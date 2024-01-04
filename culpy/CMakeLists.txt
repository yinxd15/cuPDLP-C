cmake_minimum_required (VERSION 3.0)

set (CMAKE_CXX_STANDARD 14)

set (Python_ADDITIONAL_VERSIONS 3)
find_package (PythonInterp REQUIRED)
find_package (PythonLibs REQUIRED)
if (NOT NUMPY_INCLUDES)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDES)
endif()

include_directories ("${PYTHON_INCLUDE_DIRS}")
include_directories ("${NUMPY_INCLUDES}")

set(CUPDLP_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cupdlp/")
file(GLOB CUPDLP_INCLUDE_HEADERS
        "${CUPDLP_INCLUDE_DIR}/*.h"
)
file(GLOB CUPDLP_SOURCE_FILES
        "${CUPDLP_INCLUDE_DIR}/*.c"
)

add_library (culpy SHARED
    "culpymodule.c"
    ${CUPDLP_INCLUDE_HEADERS}
    ${CUPDLP_SOURCE_FILES}
    ../interface/mps_lp.c
    ../interface/mps_lp.h
  )

set_target_properties (culpy PROPERTIES PREFIX "")

if (WIN32)
  set_target_properties (culpy PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties (culpy PROPERTIES SUFFIX ".so")
endif()

target_link_libraries (culpy "${PYTHON_LIBRARIES}")

if (${CUDA_LIBRARY-NOTFOUND})
    message(NOTICE "- CPU version PDLP")
    target_compile_definitions(culpy
            PUBLIC
            -DCUPDLP_CPU
    )
endif()