class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )
        return out


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)


        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )
        return out

    def __pow__(self, other):
        pass

    def relu(self):
        pass

    def backward(self):
        pass

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"