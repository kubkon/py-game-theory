from itertools import combinations
import numpy as np

def support_enumeration(payoff_matrix_p1, payoff_matrix_p2):
  r"""Implements support enumeration algorithm for computing all Nash
  equilibria of a bimatrix game specified by the input payoff matrices per
  player, and returns a list consisting of all Nash equilibria of the game.
  Each element of the returned list is a tuple of mixed strategies for both
  players, with the first element being the mixed strategy of the first
  player.
  
  Full theoretical description of the algorithm can be found in
  \"Algorithmic Game Theory\" by Nisan et al. (see Algorithm 3.4).
  
  IMPORTANT: The algorithm requires the game to be _nondegenerate_.
  
  Keyword arguments:
  payoff_matrix_p1 -- Payoff matrix of player 1 (np.array functionality
  required)
  payoff_matrix_p2 -- Payoff matrix of player 2 (np.array functionality
  required)
  """
  # Input params
  m, n = payoff_matrix_p1.shape
  M = set(range(m))
  N = set(range(n))
  # Output params
  msne = []
  # 1. Find set K={1,...,min{m,n}}
  K = set(range(1, min((m, n)) + 1))
  # 2. For each k in K,
  for k in K:
    # 3. Let M(k) and N(k) be sets of all k-sized subsets of M and N,
    # respectively. For each pair (I, J) such that I in M(k) and J in N(k),
    for (I, J) in [(I, J) for I in set(combinations(M, k)) \
                          for J in set(combinations(N, k))]:
      # 4. Solve for mixed strategy vectors x and y
      x = np.zeros((m, 1))
      y = np.zeros((n, 1))
      if k == 1:
        # Trivial case: pure strategies
        x[I[0]] = 1
        y[J[0]] = 1
      else:
        # Consider constraints for player 1
        v = [np.array([payoff_matrix_p2[i, j] for i in I]) for j in J]
        A = np.array([v[0]-v[p] for p in range(1, k)] + [np.ones((k, 1))])
        b = np.array((k-1)*[0] + [1])
        # Try solving matrix equation Ax = b using LU decomposition method
        try:
          solution = np.linalg.solve(A, b)
        # -- if that fails, then x cannot form Nash equilibrium
        except np.linalg.linalg.LinAlgError:
          continue
        # Create mixed strategy vector x
        solution.resize(m)
        indices = list(I)
        if len(indices) < m:
          indices += [p for p in range(m) if p not in indices]
        for (i,j) in list(map(lambda i,j: (i,j), indices, range(m))):
          x[i] = solution[j]
        # For player 2
        u = [np.array([payoff_matrix_p1[i, j] for j in J]) for i in I]
        A = np.array([u[0]-u[p] for p in range(1, k)] + [np.ones((k, 1))])
        b = np.array((k-1)*[0] + [1])
        # Try solving matrix equation Ay = b using LU decomposition method
        try:
          solution = np.linalg.solve(A, b)
        # -- if that fails, then y cannot form Nash equilibrium
        except np.linalg.linalg.LinAlgError:
          continue
        # Create mixed strategy vector y
        solution.resize(n)
        indices = list(J)
        if len(indices) < n:
          indices += [p for p in range(n) if p not in indices]
        for (i,j) in list(map(lambda i,j: (i,j), indices, range(n))):
          y[i] = solution[j]
      # Verify that (x, y) constitutes a Nash equilibrium
      # 5. Check if both x and y are nonnegative
      if (x >= 0).all() and (y >= 0).all():
        # 6. Check if best response condition is met
        # For x
        v = [np.dot(x.flatten(), payoff_matrix_p2[:,j]) for j in J]
        maximum_x = max([np.dot(x.flatten(), \
                         payoff_matrix_p2[:,n]) for n in N])
        # For y
        u = [np.dot(y.flatten(), payoff_matrix_p1[i,:]) for i in I]
        maximum_y = max([np.dot(y.flatten(), \
                         payoff_matrix_p1[m,:]) for m in M])
        # Account for numerical errors from dot product operation on floats
        if list(map(lambda el: abs(el - maximum_x) <= .0000001,  v)) \
               .count(True) == len(v) and \
           list(map(lambda el: abs(el - maximum_y) <= .0000001, u)) \
               .count(True) == len(u):
          # If the last condition is met, add (x, y) to solution list msne
          msne += [(x, y)]
  return msne

if __name__ == '__main__':
  ### Test scenario1: Equation 3.3 Nisan et al. book
  # Payoff matrices
  payoff_matrix_p1 = np.array([[3, 3], [2, 5], [0, 6]])
  payoff_matrix_p2 = np.array([[3, 2], [2, 6], [3, 1]])
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  print("MSNE for Equation 3.3 game:")
  for ne in msne:
    print("{}, {}".format(ne[0].flatten(), ne[1].flatten()))
  print()
  ### Test scenario2: Matching Pennies
  # Payoff matrices
  payoff_matrix_p1 = np.array([[-1, 1], [1, -1]])
  payoff_matrix_p2 = np.array([[1, -1], [-1, 1]])
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  print("MSNE for Matching Pennies game:")
  for ne in msne:
    print("{}, {}".format(ne[0].flatten(), ne[1].flatten()))
  print()
  ### Test scenario3: Example 2.2 Nisan et al. book
  # Payoff matrices
  payoff_matrix_p1 = np.array([[0, 3, 0], [0, 0, 3], [2, 2, 2]])
  payoff_matrix_p2 = payoff_matrix_p1.transpose()
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  print("MSNE for Example 2.2 game:")
  for ne in msne:
    print("{}, {}".format(ne[0].flatten(), ne[1].flatten()))
  print()
  ### Test scenario4: Rock-Paper-Scissors game
  # Payoff matrices
  payoff_matrix_p1 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
  payoff_matrix_p2 = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  print("MSNE for Rock-Paper-Scissors game:")
  for ne in msne:
    print("{}, {}".format(ne[0].flatten(), ne[1].flatten()))
  print()
  ### Test scenario5: Equation 3.7 Nisan et al. book
  # Payoff matrices
  payoff_matrix_p1 = np.array([[3, 3, 0], [4, 0, 1], [0, 4, 5]])
  payoff_matrix_p2 = payoff_matrix_p1.transpose()
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  print("MSNE for Equation 3.7 game:")
  for ne in msne:
    print("{}, {}".format(ne[0].flatten(), ne[1].flatten()))
  print()
  