from itertools import combinations
import numpy as np

def _test(actual, expected, description=None, debug=False):
  """Compares the numerically derived list of Nash equilibria with the
  expected (analytical) solution, and prints the result of the comparison
  to screen.
  
  Keyword arguments:
  actual -- Numerically derived list of Nash equilibria (np.array assumed)
  expected -- Expected (analytical) solution to the game
  description -- (Optional) String description of the game
  debug -- (Optional) True if print derived Nash equilibria to screen
  """
  def _round_iterable(iterable, dec_places=5):
    return map(lambda el: round(el, dec_places), iterable)
  
  actual = set([(tuple(_round_iterable(x.flatten().tolist())), tuple(_round_iterable(y.flatten().tolist())))
                for (x, y) in actual])
  expected = set([(tuple(_round_iterable(x)), tuple(_round_iterable(y))) for (x, y) in expected])
  result = "Test for game {}".format(description)
  result += " passed." if actual == expected else " failed."
  print(result)
  if debug:
    print("Derived MSNE for game {}:".format(description))
    for ne in actual:
      print("{}, {}".format(ne[0], ne[1]))
    print()

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
  payoff_matrix_p1 -- Payoff matrix of player 1 (np.array assumed)
  payoff_matrix_p2 -- Payoff matrix of player 2 (np.array assumed)
  """
  # Input params
  m, n = payoff_matrix_p1.shape
  M = range(m)
  N = range(n)
  # Output params
  msne = []
  # 1. Find set K={1,...,min{m,n}}
  K = range(1, min((m, n)) + 1)
  # 2. For each k in K,
  for k in K:
    # 3. Let M(k) and N(k) be sets of all k-sized subsets of M and N,
    # respectively. For each pair (I, J) such that I in M(k) and J in N(k),
    for (I, J) in ((I, J) for I in combinations(M, k) for J in combinations(N, k)):
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
        for (i,j) in map(lambda i,j: (i,j), indices, range(m)):
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
        for (i,j) in map(lambda i,j: (i,j), indices, range(n)):
          y[i] = solution[j]
      # Verify that (x, y) constitutes a Nash equilibrium
      # 5. Check if both x and y are nonnegative
      if all(x >= 0) and all(y >= 0):
        # 6. Check if best response condition is met
        # For x
        v = [np.dot(x.flatten(), payoff_matrix_p2[:,j]) for j in J]
        maximum_x = max([np.dot(x.flatten(), payoff_matrix_p2[:,n]) for n in N])
        # For y
        u = [np.dot(y.flatten(), payoff_matrix_p1[i,:]) for i in I]
        maximum_y = max([np.dot(y.flatten(), payoff_matrix_p1[m,:]) for m in M])
        # Account for numerical errors from dot product operation on floats
        if list(map(lambda el: abs(el - maximum_x) <= .0000001,  v)).count(True) == len(v) and \
           list(map(lambda el: abs(el - maximum_y) <= .0000001, u)).count(True) == len(u):
          # If the last condition is met, add (x, y) to solution list msne
          msne += [(x, y)]
  return msne

def vertex_enumeration(payoff_matrix_p1, payoff_matrix_p2):
  r"""Implements vertex enumeration algorithm for computing all Nash
  equilibria of a bimatrix game specified by the input payoff matrices per
  player, and returns a list consisting of all Nash equilibria of the game.
  Each element of the returned list is a tuple of mixed strategies for both
  players, with the first element being the mixed strategy of the first
  player.
  
  Full theoretical description of the algorithm can be found in
  \"Algorithmic Game Theory\" by Nisan et al. (see Algorithm 3.5).
  
  IMPORTANT: The algorithm requires the game to be _nondegenerate_, and
  payoff matrices of both players to be nonnegative and not containing
  a zero column.
  
  Keyword arguments:
  payoff_matrix_p1 -- Payoff matrix of player 1 (np.array assumed)
  payoff_matrix_p2 -- Payoff matrix of player 2 (np.array assumed)
  """
  # Input params
  m, n = payoff_matrix_p1.shape
  # Output params
  msne = []
  # 1. Find all vertices of player 1's polytope
  # Let P be the dictionary of all vertices, where key are the labels
  # corresponding to that particular vertex
  P = {}
  # Create matrices and vectors representing Player 1's polytope boundary constraints
  identity = np.identity(m, dtype=int)
  zeros_vector = np.zeros((m, 1), dtype=int)
  ones_vector = np.ones((n, 1), dtype=int)
  # For all m-combinations of the number of polytope boundary constraints,
  for rows in combinations(range(m + n), m):
    A = np.array([identity[i, :] if i < m else payoff_matrix_p2.transpose()[i % n, :] for i in rows])
    b = np.array([zeros_vector[i, :] if i < m else ones_vector[i % n, :] for i in rows])
    # Try solving matrix equation Ax = b using LU decomposition method
    try:
      x = np.linalg.solve(A, b)
    # -- if that fails, then x cannot be a vertex
    except np.linalg.linalg.LinAlgError:
      continue
    # Verify that mixed strategy vector x is a vertex
    if all(np.dot(payoff_matrix_p2.transpose(), x.flatten()) <= 1) and \
       all(np.dot(identity, x.flatten()) >= 0) and \
       not all(x == 0):
      P[rows] = x / np.sum(x)
  # 2. Find all vertices of player 2's polytope (denote by Q; key=labels)
  Q = {}
  # Create matrices and vectors representing Player 2's polytope boundary constraints
  # if the number of pure strategies is different between the players
  if n != m:
    identity = np.identity(n, dtype=int)
    zeros_vector = np.zeros((n, 1), dtype=int)
    ones_vector = np.ones((m, 1), dtype=int)
  # For all n-combinations of the number of polytope boundary constraints,
  for rows in combinations(range(n + m), n):
    A = np.array([payoff_matrix_p1[i, :] if i < m else identity[i % n, :] for i in rows])
    b = np.array([ones_vector[i, :] if i < m else zeros_vector[i % n, :] for i in rows])
    # Try solving matrix equation Ay = b using LU decomposition method
    try:
      y = np.linalg.solve(A, b)
    # -- if that fails, then y cannot be a vertex
    except np.linalg.linalg.LinAlgError:
      continue
    # Verify that mixed strategy vector y is a vertex
    if all(np.dot(payoff_matrix_p1, y.flatten()) <= 1) and \
       all(np.dot(identity, y.flatten()) >= 0) and \
       not all(y == 0):
      Q[rows] = y / np.sum(y)
  # 3. For each (x, y) if the pair is completely labeled, then (x, y) is an NE
  msne = [(P[x_labels], Q[y_labels]) for y_labels in Q for x_labels in P
          if len(set(list(x_labels) + list(y_labels))) == (m + n)]
  return msne

if __name__ == '__main__':
  ### Test scenario1: Equation 3.3 Nisan et al. book
  # Payoff matrices
  payoff_matrix_p1 = np.array([[3, 3], [2, 5], [0, 6]], dtype=int)
  payoff_matrix_p2 = np.array([[3, 2], [2, 6], [3, 1]], dtype=int)
  # Expected result
  expected = [((1.0, .0, .0), (1.0, .0)),
              ((.8, .2, .0), (2/3, 1/3)),
              ((.0, 1/3, 2/3), (1/3, 2/3))]
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  _test(msne, expected, description="Equation 3.3/support")
  # Find MSNE using vertex enumeration algorithm
  msne = vertex_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  _test(msne, expected, description="Equation 3.3/vertex")
  ### Test scenario2: Matching Pennies
  # Payoff matrices
  payoff_matrix_p1 = np.array([[-1, 1], [1, -1]], dtype=int)
  payoff_matrix_p2 = np.array([[1, -1], [-1, 1]], dtype=int)
  # Expected result
  expected = [((.5, .5), (.5, .5))]
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  _test(msne, expected, description="Matching Pennies/support")
  ### Test scenario3: Example 2.2 Nisan et al. book
  # Payoff matrices
  payoff_matrix_p1 = np.array([[0, 3, 0], [0, 0, 3], [2, 2, 2]], dtype=int)
  payoff_matrix_p2 = payoff_matrix_p1.transpose()
  # Expected result
  expected = [((.0, 1/3, 2/3), (.0, 1/3, 2/3)),
              ((.0, 2/3, 1/3), (1/3, .0, 2/3)),
              ((1/3, .0, 2/3), (.0, 2/3, 1/3))]
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  _test(msne, expected, description="Example 2.2/support")
  # Find MSNE using vertex enumeration algorithm
  msne = vertex_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  _test(msne, expected, description="Example 2.2/vertex")
  ### Test scenario4: Rock-Paper-Scissors game
  # Payoff matrices
  payoff_matrix_p1 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=int)
  payoff_matrix_p2 = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=int)
  # Expected result
  expected = [((1/3, 1/3, 1/3), (1/3, 1/3, 1/3))]
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  _test(msne, expected, description="Rock-Paper-Scissors/support")
  ### Test scenario5: Equation 3.7 Nisan et al. book
  # Payoff matrices
  payoff_matrix_p1 = np.array([[3, 3, 0], [4, 0, 1], [0, 4, 5]], dtype=int)
  payoff_matrix_p2 = payoff_matrix_p1.transpose()
  # Expected result
  expected = [((.0, .0, 1.0), (.0, .0, 1.0)),
              ((.75, .25, .0), (.75, .25, .0)),
              ((.5, .25, .25), (.5, .25, .25))]
  # Find MSNE using support enumeration algorithm
  msne = support_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  _test(msne, expected, description="Equation 3.7/support")
  # Find MSNE using vertex enumeration algorithm
  msne = vertex_enumeration(payoff_matrix_p1, payoff_matrix_p2)
  _test(msne, expected, description="Equation 3.7/vertex")
  