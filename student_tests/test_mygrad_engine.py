from mygrad.engine import Value


def test_add_forward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b

    assert c.data == 5.0


def test_multiply_forward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b

    assert c.data == 6.0