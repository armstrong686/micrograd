from micrograd.engine import Value


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

e = a * b
d = e + c
f = d.relu()

print("前向传播结果：", f)

f.backward()

print("a的梯度：", a.grad)
print("b的梯度：", b.grad)
print("c的梯度：", c.grad)