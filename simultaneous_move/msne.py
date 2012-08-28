from itertools import chain
from test import test

def linspace(begin, end, granularity):
  return map(lambda x: x/granularity, range(begin * granularity, end * granularity + 1))

def get_probability_vectors(dimension, probability_range):
  def convexify(*args):
    vector = [el for el in args]
    return tuple(vector + [1 - sum(vector)])
  vector = [[p] for p in probability_range]
  for n in range(dimension - 2):
    vector = [el + [p] for el in vector for p in probability_range]
  vector = [el for el in vector if sum(el) <= 1.0]
  return [convexify(*el) for el in vector]

def multiply_probability_vectors(vector_1, vector_2):
  return tuple(v1*v2 for v1 in vector_1 for v2 in vector_2)

def find_msne(vector_p1, utilities_p1, vector_p2, utilities_p2, error):
  result = []
  candidates = []
  for p2 in vector_p2:
    pairs = []
    values = []
    for p1 in vector_p1:
      pair = (p1, p2)
      pairs += [pair]
      values += [utilities_p1[pair]]
    maximum = max(values)
    candidates += [pair for pair in pairs if abs(values[pairs.index(pair)] - maximum) <= error]
  for can in candidates:
    pairs = [key for key in utilities_p2 if key[0] == can[0]]
    values = [utilities_p2[pair] for pair in pairs]
    index = pairs.index(can)
    value = values.pop(index)
    if max(values) - value <= error:
      result += [can]
  return result

def solve_msne_2(payoff_matrix_p1, payoff_matrix_p2, granularity=4, error=.01):
  # Construct list of probability values
  probability_range = list(linspace(0, 1, granularity))
  # Construct probability vectors for each bidder
  vector_p1 = get_probability_vectors(len(payoff_matrix_p1), probability_range)
  vector_p2 = get_probability_vectors(len(payoff_matrix_p1[0]), probability_range)
  # Construct utility dictionaries for each bidder
  probability_vectors = {(v1, v2): multiply_probability_vectors(v1, v2) for v1 in vector_p1 for v2 in vector_p2}
  payoff_vector_p1 = list(chain.from_iterable(payoff_matrix_p1))
  payoff_vector_p2 = list(chain.from_iterable(payoff_matrix_p2))
  utilities_p1 = {pair: sum(list(map(lambda x,y: x*y, probability_vectors[pair], payoff_vector_p1))) for pair in probability_vectors}
  utilities_p2 = {pair: sum(list(map(lambda x,y: x*y, probability_vectors[pair], payoff_vector_p2))) for pair in probability_vectors}
  # Find MSNE (if exist)
  msne = find_msne(vector_p1, utilities_p1, vector_p2, utilities_p2, error)
  return msne

if __name__ == '__main__':
  ### Test get_probability_vectors(...)
  # Create probability range
  probability_range = list(linspace(0, 1, 4))
  # Test output
  # Dimension of 2
  vector = set(get_probability_vectors(2, probability_range))
  expected = set(((.0, 1.0), (.25, .75), (.5, .5), (.75, .25), (1.0, .0)))
  test(expected == vector)
  # Should test higher dimensions as well...
  ### Test scenario1: Matching pennies
  # Create payoff matrices for two players
  p_matrix_p1 = [[-1, 1], [1, -1]]
  p_matrix_p2 = [[1, -1], [-1, 1]]
  # Solve for MSNE
  msne = solve_msne_2(p_matrix_p1, p_matrix_p2)
  expected = set((((.5, .5), (.5, .5)), ((.5, .5), (.5, .5))))
  test(expected == set(msne))
  ### Test scenario2: Rock-paper-scissors
  # Create payoff matrices for two players
  p_matrix_p1 = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
  p_matrix_p2 = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
  # Solve for MSNE
  msne = solve_msne_2(p_matrix_p1, p_matrix_p2, granularity=3)
  print(msne)
  ### Test scenario3: Example 4.33 from Webb's book
  # Create payoff matrices for two players
  p_matrix_p1 = [[3, 1], [0, 2]]
  p_matrix_p2 = [[2, 1], [0, 3]]
  # Solve for MSNE
  msne = solve_msne_2(p_matrix_p1, p_matrix_p2, granularity=4)
  expected = set((((.0, 1.0), (.0, 1.0)), ((.75, .25), (.25, .75)), ((1.0, .0), (1.0, .0))))
  test(expected == set(msne))
  ### Test scenario4: Example 4.38 from Webb's book
  # Create payoff matrices for two players
  p_matrix_p1 = [[10, 5, 4], [10, 5, 1]]
  p_matrix_p2 = [[0, 1, -2], [1, 0, -1]]
  # Solve for MSNE
  msne = solve_msne_2(p_matrix_p1, p_matrix_p2, granularity=4)
  print(msne)
