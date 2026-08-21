import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from mygrad.engine import Value
from mygrad.nn import MLP


# =========================
# 1. 固定随机种子
# =========================

random.seed(42)
np.random.seed(42)


# =========================
# 2. 生成月牙数据
# =========================

X, labels = make_moons(
    n_samples=100,
    noise=0.12,
    random_state=42
)

# 标准化两个输入特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将原来的0和1转换成-1和1
targets = np.where(
    labels == 1,
    1.0,
    -1.0
)


# =========================
# 3. 创建MLP
# =========================

model = MLP(
    2,
    [16, 16, 1]
)

print("模型参数量：", len(model.parameters()))


# =========================
# 4. 定义均方误差损失
# =========================

def calculate_loss(predictions, expected_targets):
    """
    计算平均均方误差：

    loss = (prediction - target) ** 2

    因为当前Value暂时没有实现减法，
    所以将prediction - target写成：
    prediction + (-target)
    """

    total_loss = Value(0.0)

    for prediction, target in zip(
        predictions,
        expected_targets
    ):
        error = prediction + (-float(target))
        sample_loss = error ** 2
        total_loss = total_loss + sample_loss

    # 用乘法代替除法
    mean_loss = total_loss * (
        1.0 / len(predictions)
    )

    return mean_loss


# =========================
# 5. 计算分类准确率
# =========================

def calculate_accuracy(
    predictions,
    expected_targets
):
    correct = 0

    for prediction, target in zip(
        predictions,
        expected_targets
    ):
        # 输出大于0，预测类别为1
        predicted_positive = prediction.data > 0

        # 目标大于0，真实类别为1
        target_positive = target > 0

        if predicted_positive == target_positive:
            correct += 1

    return correct / len(predictions)


# =========================
# 6. 正式训练
# =========================

epochs = 60
initial_learning_rate = 0.01

loss_history = []
accuracy_history = []

for epoch in range(epochs):

    # ---------- 6.1 前向传播 ----------

    predictions = []

    for sample in X:
        input_values = [
            float(sample[0]),
            float(sample[1])
        ]

        prediction = model(input_values)
        predictions.append(prediction)

    # ---------- 6.2 计算损失 ----------

    loss = calculate_loss(
        predictions,
        targets
    )

    accuracy = calculate_accuracy(
        predictions,
        targets
    )

    loss_history.append(loss.data)
    accuracy_history.append(accuracy)

    # ---------- 6.3 梯度清零 ----------

    model.zero_grad()

    # ---------- 6.4 反向传播 ----------

    loss.backward()

    # ---------- 6.5 学习率逐渐降低 ----------

    progress = epoch / max(epochs - 1, 1)

    learning_rate = (
        initial_learning_rate
        * (1.0 - 0.5 * progress)
    )

    # ---------- 6.6 更新所有参数 ----------

    for parameter in model.parameters():
        parameter.data -= (
            learning_rate * parameter.grad
        )

    # ---------- 6.7 打印训练状态 ----------

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.data:.4f} | "
            f"Accuracy: {accuracy:.2%} | "
            f"Learning rate: "
            f"{learning_rate:.5f}"
        )


# =========================
# 7. 不构建计算图的快速预测
# =========================

def predict_number(model_object, sample):
    """
    只使用参数的数值进行预测，不创建Value计算图。

    绘制大量网格点时使用这个函数，
    否则速度会非常慢。
    """

    values = [
        float(sample[0]),
        float(sample[1])
    ]

    for layer in model_object.layers:
        layer_outputs = []

        for neuron in layer.neurons:
            activation = neuron.b.data

            for weight, input_value in zip(
                neuron.w,
                values
            ):
                activation += (
                    weight.data * input_value
                )

            if neuron.nonlin:
                activation = max(
                    0.0,
                    activation
                )

            layer_outputs.append(activation)

        values = layer_outputs

    return values[0]


# =========================
# 8. 生成分类边界网格
# =========================

x_min = X[:, 0].min() - 0.6
x_max = X[:, 0].max() + 0.6

y_min = X[:, 1].min() - 0.6
y_max = X[:, 1].max() + 0.6

grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid_points = np.column_stack(
    (
        grid_x.ravel(),
        grid_y.ravel()
    )
)

grid_predictions = np.array(
    [
        predict_number(model, point)
        for point in grid_points
    ]
)

grid_classes = (
    grid_predictions > 0
).astype(int)

grid_classes = grid_classes.reshape(
    grid_x.shape
)


# =========================
# 9. 绘制结果
# =========================

figure, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

# ---------- 左图：分类边界 ----------

axes[0].contourf(
    grid_x,
    grid_y,
    grid_classes,
    levels=[-0.5, 0.5, 1.5],
    cmap="coolwarm",
    alpha=0.35
)

axes[0].scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="coolwarm",
    edgecolors="black",
    s=45
)

axes[0].set_title(
    "MyGrad Moon Classification"
)

axes[0].set_xlabel(
    "Standardized feature 1"
)

axes[0].set_ylabel(
    "Standardized feature 2"
)

# ---------- 右图：损失曲线 ----------

axes[1].plot(
    range(epochs),
    loss_history,
    color="blue",
    linewidth=2
)

axes[1].set_title(
    "Training Loss"
)

axes[1].set_xlabel(
    "Epoch"
)

axes[1].set_ylabel(
    "Mean Squared Error"
)

axes[1].grid(
    True,
    alpha=0.3
)

figure.tight_layout()


# =========================
# 10. 保存并显示图片
# =========================

output_path = (
    Path(__file__).resolve().parent
    / "moons_training_result.png"
)

figure.savefig(
    output_path,
    dpi=160,
    bbox_inches="tight"
)

print("\n训练完成")
print(
    f"最终损失：{loss_history[-1]:.4f}"
)
print(
    f"最终准确率："
    f"{accuracy_history[-1]:.2%}"
)
print(
    f"图片已保存到：{output_path}"
)

plt.show()