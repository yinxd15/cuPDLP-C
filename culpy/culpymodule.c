#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"

#include "../cupdlp/cupdlp.h"
#include "../interface/mps_lp.h"


// min    c'x
// s.t.   A[:nEqs, :]x + b[:nEqs] == 0
//        A[nEqs:, :]x + b[nEqs:] >= 0
//        x free
// where A is given in CSC format.
int wrapper(
    cupdlp_int nRows,
    cupdlp_int nCols,
    cupdlp_int nnz,
    cupdlp_int nEqs,
    cupdlp_int *colMatBeg,
    cupdlp_int *colMatIdx,
    cupdlp_float *colMatElem,
    cupdlp_float *b,
    cupdlp_float *c,
    cupdlp_float *lower,
    cupdlp_float *upper,
    cupdlp_float *x,
    cupdlp_float *y,
    cupdlp_bool *ifChangeIntParam,
    cupdlp_int *intParam,
    cupdlp_bool *ifChangeFloatParam,
    cupdlp_float *floatParam,
    cupdlp_float *pcost,
    cupdlp_float *solve_time,
    cupdlp_float *setup_time,
    cupdlp_int *num_iters) {
  CUPDLPproblem *prob = cupdlp_NULL;
  CUPDLPwork *w = cupdlp_NULL;
  w = (CUPDLPwork *)calloc(1, sizeof(CUPDLPwork));

  // Cvxpy does not provides bounds explicitly. We allow bounds to be empty.
  if (lower == cupdlp_NULL)
  {
    lower = (cupdlp_float *)calloc(nCols, sizeof(cupdlp_float));
    for (int i = 0; i < nCols; i++)
    {
      lower[i] = -INFINITY;
    }
  }
  if (upper == cupdlp_NULL)
  {
    upper = (cupdlp_float *)calloc(nCols, sizeof(cupdlp_float));
    for (int i = 0; i < nCols; i++)
    {
      upper[i] = INFINITY;
    }
  }

#if !(CUPDLP_CPU)
  cupdlp_float cuda_prepare_time = getTimeStamp();
  CHECK_CUSPARSE(cusparseCreate(&w->cusparsehandle));
  CHECK_CUBLAS(cublasCreate(&w->cublashandle));
  cuda_prepare_time = getTimeStamp() - cuda_prepare_time;
#endif

  problem_create(&prob);

  CUPDLPcsc *csc_cpu = cupdlp_NULL;
  // prepare csc matrix
  csc_create(&csc_cpu);
  csc_cpu->nRows = nRows;
  csc_cpu->nCols = nCols;
  csc_cpu->nMatElem = nnz;
  csc_cpu->colMatBeg = colMatBeg;
  csc_cpu->colMatIdx = colMatIdx;
  csc_cpu->colMatElem = colMatElem;
#if !(CUPDLP_CPU)
  csc_cpu->cuda_csc = NULL;
#endif


  CUPDLPscaling *scaling = (CUPDLPscaling *)cupdlp_malloc(sizeof(CUPDLPscaling));
  // scaling
  cupdlp_float scaling_time = getTimeStamp();
  Init_Scaling(scaling, nCols, nRows, c, b);
  PDHG_Scale_Data_cuda(csc_cpu, 1, scaling, c, lower, upper, b);
  scaling_time = getTimeStamp() - scaling_time;

  // ?
  cupdlp_float alloc_matrix_time = 0.0;
  cupdlp_float copy_vec_time = 0.0;

  problem_alloc(prob, nRows, nCols, nEqs, c, 0, 1, csc_cpu, CSC,
                CSR_CSC, b, lower, upper,
                &alloc_matrix_time, &copy_vec_time);

  // ?
  w->problem = prob;
  w->scaling = scaling;
  PDHG_Alloc(w);
  w->timers->dScalingTime = scaling_time;
  w->timers->dPresolveTime = 0.0;

  CUPDLP_COPY_VEC(w->rowScale, scaling->rowScale, cupdlp_float, nRows);
  CUPDLP_COPY_VEC(w->colScale, scaling->colScale, cupdlp_float, nCols);

#if !(CUPDLP_CPU)
  w->timers->AllocMem_CopyMatToDeviceTime += alloc_matrix_time;
  w->timers->CopyVecToDeviceTime += copy_vec_time;
  w->timers->CudaPrepareTime = cuda_prepare_time;
#endif

  // solve
  cupdlp_retcode status = LP_SolvePDHG_New(w, ifChangeIntParam, intParam, ifChangeFloatParam, floatParam,
               x, nCols, y, cupdlp_NULL);
  *num_iters = w->timers->nIter;
  *solve_time = w->timers->dSolvingTime + w->timers->dPresolveTime + w->timers->dScalingTime;
  *pcost = w->resobj->dPrimalObj;
#if !(CUPDLP_CPU)
  *setup_time = (
    w->timers->AllocMem_CopyMatToDeviceTime 
    + w->timers->CopyVecToDeviceTime
    + w->timers->CopyVecToHostTime
  );
#else
  *setup_time = 0.0;
#endif

exit_cleanup:

  if (scaling)
  {
    scaling_clear(scaling);
  }
  // free memory
  problem_clear(prob);
  return status;
}

void np2arr_int(PyArrayObject *obj, cupdlp_int **arr, cupdlp_int length) {
  *arr = calloc(length, sizeof(cupdlp_int));
  for (int i = 0; i < length; i++)
  {
    (*arr)[i] = *(cupdlp_int *)PyArray_GETPTR1(obj, i);
  }
}

void np2arr_float(PyArrayObject *obj, cupdlp_float **arr, cupdlp_int length) {
  *arr = calloc(length, sizeof(cupdlp_float));
  for (int i = 0; i < length; i++)
  {
    (*arr)[i] = *(cupdlp_float *)PyArray_GETPTR1(obj, i);
  }
}

void arr2np_float(cupdlp_float *arr, PyArrayObject **obj, cupdlp_int length) {
  npy_intp dims[1] = {length};
  *obj = PyArray_SimpleNew(1, &dims, NPY_FLOAT64);
  for (int i = 0; i < length; i++)
  {
    *(cupdlp_float *)PyArray_GETPTR1(*obj, i) = arr[i];
  }
}

static PyObject *culpy_solve(PyObject *self, PyObject *args) {
  cupdlp_int nRows;
  cupdlp_int nCols;
  cupdlp_int nnz;
  cupdlp_int nEqs;
  PyArrayObject *colMatBegObj;
  PyArrayObject *colMatIdxObj;
  PyArrayObject *colMatElemObj;
  PyArrayObject *bObj;
  PyArrayObject *cObj;
  PyObject *paras;
  cupdlp_int *colMatBeg;
  cupdlp_int *colMatIdx;
  cupdlp_float *colMatElem;
  cupdlp_float *b;
  cupdlp_float *c;

  // TODO(yinxd 20240102): select `i` according to the type of cupdlp_int
  if (!PyArg_ParseTuple(args, "iiiiOOOOOO!",
    &nRows,
    &nCols,
    &nnz,
    &nEqs,
    &colMatBegObj,
    &colMatIdxObj,
    &colMatElemObj,
    &bObj,
    &cObj,
    &PyDict_Type,
    &paras))
    return NULL;


  // set solver parameters
  cupdlp_bool ifChangeIntParam[N_INT_USER_PARAM] = {false};
  cupdlp_int intParam[N_INT_USER_PARAM] = {0};
  cupdlp_bool ifChangeFloatParam[N_FLOAT_USER_PARAM] = {false};
  cupdlp_float floatParam[N_FLOAT_USER_PARAM] = {0.0};

  // iterate over paras
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(paras, &pos, &key, &value)) {

    const char* keyStr = PyUnicode_AsUTF8(key);

    if (strcmp(keyStr, "nIterLim") == 0) {
      ifChangeIntParam[N_ITER_LIM] = true;
      intParam[N_ITER_LIM] = PyLong_AsLong(value);
    } else if (strcmp(keyStr, "ifScaling") == 0) {
      ifChangeIntParam[IF_SCALING] = true;
      intParam[IF_SCALING] = PyLong_AsLong(value);
    } else if (strcmp(keyStr, "iScalingMethod") == 0) {
      ifChangeIntParam[I_SCALING_METHOD] = true;
      intParam[I_SCALING_METHOD] = PyLong_AsLong(value);
    } else if (strcmp(keyStr, "eLineSearchMethod") == 0) {
      ifChangeIntParam[E_LINE_SEARCH_METHOD] = true;
      intParam[E_LINE_SEARCH_METHOD] = PyLong_AsLong(value);
    } else if (strcmp(keyStr, "dScalingLimit") == 0) {
      ifChangeFloatParam[D_SCALING_LIMIT] = true;
      floatParam[D_SCALING_LIMIT] = PyFloat_AsDouble(value);
    } else if (strcmp(keyStr, "dPrimalTol") == 0) {
      ifChangeFloatParam[D_PRIMAL_TOL] = true;
      floatParam[D_PRIMAL_TOL] = PyFloat_AsDouble(value);
    } else if (strcmp(keyStr, "dDualTol") == 0) {
      ifChangeFloatParam[D_DUAL_TOL] = true;
      floatParam[D_DUAL_TOL] = PyFloat_AsDouble(value);
    } else if (strcmp(keyStr, "dGapTol") == 0) {
      ifChangeFloatParam[D_GAP_TOL] = true;
      floatParam[D_GAP_TOL] = PyFloat_AsDouble(value);
    } else if (strcmp(keyStr, "dTimeLim") == 0) {
      ifChangeFloatParam[D_TIME_LIM] = true;
      floatParam[D_TIME_LIM] = PyFloat_AsDouble(value);
    } else if (strcmp(keyStr, "eRestartMethod") == 0) {
      ifChangeIntParam[E_RESTART_METHOD] = true;
      intParam[E_RESTART_METHOD] = PyLong_AsLong(value);
    } else if (strcmp(keyStr, "ifRuizScaling") == 0) {
      ifChangeIntParam[IF_RUIZ_SCALING] = true;
      intParam[IF_RUIZ_SCALING] = PyLong_AsLong(value);
    } else if (strcmp(keyStr, "ifL2Scaling") == 0) {
      ifChangeIntParam[IF_L2_SCALING] = true;
      intParam[IF_L2_SCALING] = PyLong_AsLong(value);
    } else if (strcmp(keyStr, "ifPcScaling") == 0) {
      ifChangeIntParam[IF_PC_SCALING] = true;
      intParam[IF_PC_SCALING] = PyLong_AsLong(value);
    } else if (strcmp(keyStr, "nLogInt") == 0) {
      ifChangeIntParam[N_LOG_INTERVAL] = true;
      intParam[N_LOG_INTERVAL] = PyLong_AsLong(value);
    } else if (strcmp(keyStr, "ifPresolve") == 0) {
      ifChangeIntParam[IF_PRESOLVE] = true;
      intParam[IF_PRESOLVE] = PyLong_AsLong(value);
    } else {
      cupdlp_printf("Unknown parameter: %s\n", keyStr);
    }
  }

  np2arr_int(colMatBegObj, &colMatBeg, nCols + 1);
  np2arr_int(colMatIdxObj, &colMatIdx, nnz);
  np2arr_float(colMatElemObj, &colMatElem, nnz);
  np2arr_float(bObj, &b, nRows);
  np2arr_float(cObj, &c, nCols);

  cupdlp_float *x = cupdlp_NULL;
  cupdlp_float *y = cupdlp_NULL;
  x = (cupdlp_float *)calloc(nCols, sizeof(cupdlp_float));
  y = (cupdlp_float *)calloc(nRows, sizeof(cupdlp_float));

  cupdlp_float pcost = 0.0;
  cupdlp_float solve_time = 0.0;
  cupdlp_float setup_time = 0.0;
  cupdlp_int num_iters = 0;

  cupdlp_int status = wrapper(
    nRows, nCols, nnz, nEqs, colMatBeg, colMatIdx, colMatElem, 
    b, c, cupdlp_NULL, cupdlp_NULL, x, y,
    ifChangeIntParam, intParam, ifChangeFloatParam, floatParam,
    &pcost, &solve_time, &setup_time, &num_iters);

  PyObject *xObj = cupdlp_NULL;
  PyObject *yObj = cupdlp_NULL;
  arr2np_float(x, &xObj, nCols);
  arr2np_float(y, &yObj, nRows);

  PyObject *d = PyDict_New();

  PyDict_SetItem(d, PyUnicode_FromString("status"), PyLong_FromLong(status));
  PyDict_SetItem(d, PyUnicode_FromString("primal_vars"), xObj);
  PyDict_SetItem(d, PyUnicode_FromString("dual_vars"), yObj);
  PyDict_SetItem(d, PyUnicode_FromString("pcost"), PyFloat_FromDouble(pcost));
  PyDict_SetItem(d, PyUnicode_FromString("solve_time"), PyFloat_FromDouble(solve_time));
  PyDict_SetItem(d, PyUnicode_FromString("setup_time"), PyFloat_FromDouble(setup_time));
  PyDict_SetItem(d, PyUnicode_FromString("num_iters"), PyLong_FromLong(num_iters));

  return d;
}

static PyMethodDef culpy_methods[] = {
    {"solve", culpy_solve, METH_VARARGS, "Solve a LP model."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef culpy_module = {
    PyModuleDef_HEAD_INIT,
    "culpy",
    NULL,
    -1,
    culpy_methods
};

PyObject *PyInit_culpy(void) {
  import_array();
  return PyModule_Create(&culpy_module);
}

