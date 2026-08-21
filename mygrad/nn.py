import random

from .engine import Value


class Module:
    """所有神经网络模块的基类"""

    def parameters(self):
        """返回模块中的全部可训练参数"""
        return []

    def zero_grad(self):
        """将所有参数的梯度清零"""
        for parameter in self.parameters():
            parameter.grad = 0.0


class Neuron(Module):
    """单个神经元"""

    def __init__(self, nin, nonlin=True):
        """
        nin：输入特征数量
        nonlin：是否使用ReLU激活函数
        """

        # 每个输入对应一个权重
        self.w = [
            Value(random.uniform(-1, 1))
            for _ in range(nin)
        ]

        # 一个神经元有一个偏置
        self.b = Value(0.0)

        # 是否使用非线性激活函数
        self.nonlin = nonlin

    def __call__(self, x):
        """执行神经元的前向计算"""

        if len(x) != len(self.w):
            raise ValueError(
                f"输入维度为{len(x)}，"
                f"但神经元需要{len(self.w)}个输入"
            )

        # z = w1*x1 + w2*x2 + ... + b
        activation = self.b

        for weight, input_value in zip(self.w, x):
            activation = (
                activation + weight * input_value
            )

        # 隐藏层使用ReLU，输出层不使用
        if self.nonlin:
            return activation.relu()

        return activation

    def parameters(self):
        """返回当前神经元的所有权重和偏置"""
        return self.w + [self.b]

    def __repr__(self):
        activation_name = (
            "ReLUNeuron"
            if self.nonlin
            else "LinearNeuron"
        )

        return (
            f"{activation_name}("
            f"input_features={len(self.w)})"
        )


class Layer(Module):
    """由多个神经元组成的一层网络"""

    def __init__(self, nin, nout, nonlin=True):
        """
        nin：每个神经元的输入数量
        nout：这一层的神经元数量
        """

        self.neurons = [
            Neuron(nin, nonlin=nonlin)
            for _ in range(nout)
        ]

    def __call__(self, x):
        """让输入依次经过当前层中的所有神经元"""

        outputs = [
            neuron(x)
            for neuron in self.neurons
        ]

        # 如果只有一个输出，直接返回Value
        if len(outputs) == 1:
            return outputs[0]

        # 多个输出时返回Value列表
        return outputs

    def parameters(self):
        """返回当前层全部神经元的参数"""

        parameters = []

        for neuron in self.neurons:
            parameters.extend(neuron.parameters())

        return parameters

    def __repr__(self):
        return (
            f"Layer(neurons={self.neurons})"
        )


class MLP(Module):
    """多层感知机"""

    def __init__(self, nin, nouts):
        """
        nin：最初的输入维度

        nouts：每一层的神经元数量，例如：
        [16, 16, 1]
        """

        if not nouts:
            raise ValueError("nouts不能为空")

        # 例如：[2, 16, 16, 1]
        layer_sizes = [nin] + list(nouts)

        self.layers = []

        for index in range(len(nouts)):
            input_size = layer_sizes[index]
            output_size = layer_sizes[index + 1]

            # 最后一层不使用ReLU
            is_last_layer = index == len(nouts) - 1

            layer = Layer(
                nin=input_size,
                nout=output_size,
                nonlin=not is_last_layer
            )

            self.layers.append(layer)

    def __call__(self, x):
        """让输入依次经过所有网络层"""

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        """返回整个MLP的全部参数"""

        parameters = []

        for layer in self.layers:
            parameters.extend(layer.parameters())

        return parameters

    def __repr__(self):
        return (
            f"MLP(layers={self.layers})"
        )