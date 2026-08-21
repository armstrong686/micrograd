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
def test_backward_graph():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    result = (a * b + c).relu()
    result.backward()

    assert result.data == 4.0
    assert result.grad == 1.0

    assert a.grad == -3.0
    assert b.grad == 2.0
    assert c.grad == 1.0


def test_power_backward():
    x = Value(3.0)

    result = x ** 2
    result.backward()

    assert result.data == 9.0
    assert x.grad == 6.0


def test_relu_negative():
    x = Value(-5.0)

    result = x.relu()
    result.backward()

    assert result.data == 0.0
    assert x.grad == 0.0


def test_gradient_accumulation():
    x = Value(3.0)

    result = x * x + x
    result.backward()

    assert result.data == 12.0
    assert x.grad == 7.0