from test import test
import numpy as np
import itertools as it

def support_enumeration(payoff_matrix_p1, payoff_matrix_p2):
  # Assumes matrices are nondegenerate!
  # Input params
  m, n = payoff_matrix_p1.shape
  M = set(range(m))
  N = set(range(n))
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
      print(payoff_matrix_p1[0][0])
  return None

def lemke_howson(payoff_matrix_p1, payoff_matrix_p2):
  # FIXME ---
  # Create symmetric matrices if not symmetric already
  payoff_matrix = payoff_matrix_p1
  # Verify that matrix is nondegenerate
  # Find all vertices of the polytype
  # ---
  # Proceed with the algorithm
  # 1. Fix a strategy: row dimension
  strategy = payoff_matrix.shape[0]
  # 2. Start at zeroth vector
  init_vector = np.zeros((strategy, 1))
  # 3. Find another vertex
  return None

if __name__ == '__main__':
  ### Test scenario1: Example 2.2 Nisan's et al. book
  # Payoff matrices
  payoff_matrix_p1 = payoff_matrix_p2 = np.array([[0, 3, 0], [0, 0, 3], [2, 2, 2]])
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  