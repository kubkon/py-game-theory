def test(condition):
  try:
    assert condition
  except AssertionError as e:
    print("Test failed")
  else:
    print("Test successful")
