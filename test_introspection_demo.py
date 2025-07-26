"""Simple test file to demonstrate SubagentStop hook introspection."""


def add_numbers(a, b):
    """Add two numbers and return the result.

    # TODO: Add type hints to function parameters and return value
    """
    return a + b


def test_add_numbers():
    """Test the add_numbers function."""
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
    print("All tests passed!")


if __name__ == "__main__":
    # Run the test
    test_add_numbers()

    # Demonstrate the function
    result = add_numbers(10, 20)
    print(f"10 + 20 = {result}")
