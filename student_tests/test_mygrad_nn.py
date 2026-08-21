from mygrad.engine import Value
from mygrad.nn import Neuron, Layer, MLP


def test_neuron_parameters():
    neuron = Neuron(2)

    # 两个权重加一个偏置
    assert len(neuron.parameters()) == 3


def test_layer_parameters():
    layer = Layer(
        nin=2,
        nout=3
    )

    # 三个神经元，每个有2个权重和1个偏置
    assert len(layer.parameters()) == 9


def test_mlp_parameters():
    model = MLP(
        2,
        [16, 16, 1]
    )

    # 第一层：16 × (2 + 1) = 48
    # 第二层：16 × (16 + 1) = 272
    # 输出层：1 × (16 + 1) = 17
    # 总数：48 + 272 + 17 = 337
    assert len(model.parameters()) == 337


def test_mlp_forward():
    model = MLP(
        2,
        [4, 1]
    )

    output = model([2.0, -1.0])

    assert isinstance(output, Value)


def test_mlp_backward():
    model = MLP(
        2,
        [1]
    )

    output = model([2.0, -1.0])

    model.zero_grad()
    output.backward()

    # 最后一项参数是输出神经元的偏置
    output_bias = model.parameters()[-1]

    assert output_bias.grad == 1.0


def test_zero_grad():
    model = MLP(
        2,
        [4, 1]
    )

    for parameter in model.parameters():
        parameter.grad = 10.0

    model.zero_grad()

    assert all(
        parameter.grad == 0.0
        for parameter in model.parameters()
    )