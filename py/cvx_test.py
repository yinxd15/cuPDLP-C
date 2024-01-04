# Import packages.
import cvxpy as cp
import numpy as np

# Generate a random non-trivial linear program.
m = 15
n = 10
np.random.seed(1)
s0 = np.random.randn(m)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b])
prob.solve(solver='CUPDLP', dPrimalTol=1e-8, dDualTol=1e-8, dGapTol=1e-8, nIterLim=1234, dTimeLim=42.31)

xv = x.value

prob.solve(solver='SCIPY')

xv2 = x.value

print('Primal residual by CUPDLP:', b - A @ xv)
print('Primal residual by Scipy:', b - A @ xv2)
print('Primal cost by CUPDLP:', c @ xv)
print('Primal cost by Scipy:', c @ xv2)
print('2-norm of difference of primal sol:', np.linalg.norm(xv - xv2))