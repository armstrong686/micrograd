from mygrad.nn import MLP


model = MLP(
    2,
    [16, 16, 1]
)

print("网络结构：")
print(model)

print("\n参数总数：")
print(len(model.parameters()))

x = [2.0, -1.0]

prediction = model(x)

print("\n输入：")
print(x)

print("\n网络输出：")
print(prediction)

model.zero_grad()
prediction.backward()

print("\n前5个参数的梯度：")

for parameter in model.parameters()[:5]:
    print(parameter.grad)