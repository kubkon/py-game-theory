from numpy import array, linspace
from itertools import chain

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
  p_matrix_p1 = [[1, 4, 2], [4, 0, 4]]#, [2, 3, 5]]
  p_matrix_p2 = [[3, 2, 2], [0, 3, 1]]#, [5, 4, 6]]
  # Solve for PSNE
  psne = solve_psne_2(p_matrix_p1, p_matrix_p2)
  test(psne == [(2, 2)])
