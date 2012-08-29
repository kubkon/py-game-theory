from test import test
import numpy as np
import itertools as it

def support_enumeration(payoff_matrix_p1, payoff_matrix_p2):
  # Assumes matrices are nondegenerate!
  # Input params
  m, n = payoff_matrix_p1.shape
  M = set(range(m))
  N = set(range(n))
  # Output params
  solutions = []
  # 1. Find set K={1,...,min{m,n}}
  K = set(range(1, min((m, n)) + 1))
  # 2. For each k in K
  for k in K:
    # 3. Find all k-sized subsets of M and N
    M_k, N_k = set(it.combinations(M, k)), set(it.combinations(N, k))
    # 4. Find set of all possible pairs (I, J) such that I in M_k, and J in N_k
    I_J_pairs = [(I, J) for I in M_k for J in N_k]
    # 5. For each I_J pair in I_J_pairs
    for I_J in I_J_pairs:
      # 6. Solve the equations
      I = I_J[0]
      J = I_J[1]
      x = np.zeros((m, 1))
      y = np.zeros((n, 1))
      if k == 1:
        # Create probability vectors x and y
        x[I[0]] = 1
        y[J[0]] = 1
      else:
        # For player 1
        v = [np.array([payoff_matrix_p2[i, j] for i in I]) for j in J]
        A = np.array([v[0]-v[p] for p in range(1, k)] + [np.ones((k, 1))])
        b = np.array((k-1)*[0] + [1])
        solution = np.linalg.solve(A, b)
        # Create probability vector x
        solution.resize(m)
        I = list(I)
        if len(I) < m:
          for p in range(m):
            if p not in I:
              I += [p]
        count = 0
        for i in I:
          x[i] = solution[count]
          count += 1
        # For player 2
        u = [np.array([payoff_matrix_p1[i, j] for j in J]) for i in I]
        A = np.array([u[0]-u[p] for p in range(1, k)] + [np.ones((k, 1))])
        b = np.array((k-1)*[0] + [1])
        solution = np.linalg.solve(A, b)
        # Create probability vector y
        solution.resize(n)
        J = list(J)
        if len(J) < n:
          for p in range(n):
            if p not in J:
              J += [p]
        count = 0
        for j in J:
          y[j] = solution[count]
          count += 1
      # Perform solution checks
      # 7. Check if both x and y are nonnegative
      if (x >= 0).all() and (y >= 0).all():
        # 8. Check if best response condition is met
        # For x
        v = [np.dot(x.flatten(), payoff_matrix_p2[:,j]) for j in I_J[1]]
        maximum_x = max([np.dot(x.flatten(), payoff_matrix_p2[:,n]) for n in N])
        # For y
        u = [np.dot(y.flatten(), payoff_matrix_p1[i,:]) for i in I_J[0]]
        maximum_y = max([np.dot(y.flatten(), payoff_matrix_p1[m,:]) for m in M])
        if v.count(maximum_x) == len(v) and u.count(maximum_y) == len(u):
          solutions += [(x, y)]
  return solutions

if __name__ == '__main__':
  ### Test scenario1: Example 3.3 Nisan et al. book
  # Payoff matrices
  payoff_matrix_p1 = np.array([[3, 3], [2, 5], [0, 6]])
  payoff_matrix_p2 = np.array([[3, 2], [2, 6], [3, 1]])  
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  print(msne)
  