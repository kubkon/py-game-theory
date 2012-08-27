from numpy import array, linspace

def transpose(payoff_matrix):
  return array(payoff_matrix).transpose().tolist()

def get_best_responses(payoff_matrix):
  # Select argmax from each row, and return the result as a list
  best_responses = []
  for row in payoff_matrix:
    count = 0
    for el in row:
      maximum = max(row)
      if el == maximum:
        best_responses += [(payoff_matrix.index(row), count)]
      count += 1
  return best_responses

def solve_psne_2(payoff_matrix_p1, payoff_matrix_p2):
  # Transpose payoff matrix for player 1, and get best responses
  indices_p1 = get_best_responses(transpose(payoff_matrix_p1))
  # Swap values in each pair of indices (i.e., reverse transposition)
  indices_p1 = list(map(lambda x: (x[1], x[0]), indices_p1))
  # Get best responses for player 2
  indices_p2 = get_best_responses(payoff_matrix_p2)
  # Return PSNE (if exist)
  psne = []
  if len(indices_p1) >= len(indices_p2):
    psne = [el for el in indices_p1 if el in indices_p2]
  else:
    psne = [el for el in indices_p2 if el in indices_p1]
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
  test(set(psne) == set(((1, 1),)))
  ### Test scenario2: Matching pennies
  # Create payoff matrices for two players
  p_matrix_p1 = [[-1, 1], [1, -1]]
  p_matrix_p2 = [[1, -1], [-1, 1]]
  # Solve for PSNE
  psne = solve_psne_2(p_matrix_p1, p_matrix_p2)
  test(psne == [])
  ### Test scenario3: Example 4.16 from Webb's book
  # Create payoff matrices for two players
  p_matrix_p1 = [[1, 4, 2], [4, 0, 4], [2, 3, 5]]
  p_matrix_p2 = [[3, 2, 2], [0, 3, 1], [5, 4, 6]]
  # Solve for PSNE
  psne = solve_psne_2(p_matrix_p1, p_matrix_p2)
  test(set(psne) == set(((2, 2),)))
  ### Test scenario4: Exercise 4.3 from Webb's book
  # Create payoff matrices for two players
  p_matrix_p1 = [[10, 5, 4], [10, 5, 1]]
  p_matrix_p2 = [[0, 1, -2], [1, 0, -1]]
  # Solve for PSNE
  psne = solve_psne_2(p_matrix_p1, p_matrix_p2)
  test(set(psne) == set(((1, 0), (0, 1))))
