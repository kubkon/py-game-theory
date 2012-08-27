from numpy import array, linspace
from itertools import chain

def operator(*args):
  vector = [el for el in args]
  return vector + [1 - sum(vector)]

def transpose(payoff_matrix):
  return array(payoff_matrix).transpose().tolist()

def get_best_responses(payoff_matrix):
  # Select argmax from each row, and return the result as a list
  return list(map(lambda x: (payoff_matrix.index(x), x.index(max(x))), payoff_matrix))

def solve_psne_2(payoff_matrix_p1, payoff_matrix_p2):
  # Transpose payoff matrix for player 1, and get best responses
  indices_p1 = get_best_responses(transpose(payoff_matrix_p1))
  # Swap values in each pair of indices (i.e., reverse transposition)
  indices_p1 = list(map(lambda x: (x[1], x[0]), indices_p1))
  # Get best responses for player 2
  indices_p2 = get_best_responses(payoff_matrix_p2)
  # Return PSNE (if exist)
  matched_responses = list(map(lambda x, y: x == y, indices_p1, indices_p2))
  psne = [indices for indices in indices_p1 if matched_responses[indices_p1.index(indices)] == True]
  return psne

def solve_msne_2(payoff_matrix_p1, payoff_matrix_p2, granularity=5):
  # Construct list of probability values
  probability_range = linspace(0.0, 1.0, granularity).tolist()
  # Construct dict of probability vectors
  row_dimension = len(payoff_matrix_p1)
  col_dimension = len(payoff_matrix_p1[0])
  hmm = [[p] for p in probability_range]
  for m in range(row_dimension - 2):
    hmm = [el + [p] for el in hmm for p in probability_range]
  hmm = [el for el in hmm if sum(el) <= 1.0]
  print(hmm)
  vector = [operator(*el) for el in hmm]
  print(vector)
  # Other...
  # probability_vectors = {(p,q): [p*q, p*(1-q), (1-p)*q, (1-p)*(1-q)] for p in probability_range for q in probability_range}
  # payoff_vector_1 = list(chain.from_iterable(payoff_matrix_p1))
  # payoff_vector_2 = list(chain.from_iterable(payoff_matrix_p2))
  # utilities_1 = {pair: sum(list(map(lambda x, y: x*y, probability_vectors[pair], payoff_vector_1))) for pair in probability_vectors}
  # utilities_2 = {pair: sum(list(map(lambda x, y: x*y, probability_vectors[pair], payoff_vector_2))) for pair in probability_vectors}
  # # print(utilities_1)
  # # print(utilities_2)
  # matched = list(map(lambda x, y: (x[0], abs(x[1] - y[1])), sorted(utilities_1.items()), sorted(utilities_2.items())))
  # minimum = min(matched, key=lambda x: x[1])
  # print(sorted([x for x in matched if x[1] == minimum[1]], key=lambda x: x[0]))
  return None

def test(condition):
  try:
    assert condition
  except AssertionError as e:
    print("Test failed")
  else:
    print("Test successful")

if __name__ == '__main__':
  ### Test scenario1: Prisoner's dilemma
  # Create payoff matrices for two players
  p_matrix_p1 = [[-1, -10], [0, -3]]
  p_matrix_p2 = [[-1, 0], [-10, -3]]
  # Solve for PSNE
  psne = solve_psne_2(p_matrix_p1, p_matrix_p2)
  test(psne == [(1, 1)])
  ### Test scenario2: Matching pennies
  # Create payoff matrices for two players
  p_matrix_p1 = [[-1, 1], [1, -1]]
  p_matrix_p2 = [[1, -1], [-1, 1]]
  # Solve for PSNE
  psne = solve_psne_2(p_matrix_p1, p_matrix_p2)
  test(psne == [])
  ### Test scenario3: Example 4.16 from Carter's book
  # Create payoff matrices for two players
  p_matrix_p1 = [[1, 4, 2], [4, 0, 4], [2, 3, 5]]
  p_matrix_p2 = [[3, 2, 2], [0, 3, 1], [5, 4, 6]]
  # Solve for PSNE
  psne = solve_psne_2(p_matrix_p1, p_matrix_p2)
  test(psne == [(2, 2)])
  # Solve for MSNE
  msne = solve_msne_2(p_matrix_p1, p_matrix_p2)
  