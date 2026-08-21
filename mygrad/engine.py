class Value:
    def __init__(self, data, _children=(), _op=""):
        # 当前节点保存的数值
        self.data = data

        # 最终结果对当前节点的梯度
        self.grad = 0.0

        # 当前节点对应的局部反向传播函数
        self._backward = lambda: None

        # 当前节点依赖的父节点
        self._prev = set(_children)

        # 产生当前节点的运算符
        self._op = _op

    def __add__(self, other):
        """加法：z = x + y"""
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            # dz/dx = 1，dz/dy = 1
            # 再乘以上游梯度out.grad
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """乘法：z = x * y"""
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            # dz/dx = y
            self.grad += other.data * out.grad

            # dz/dy = x
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        """幂运算：z = x ** n"""
        assert isinstance(other, (int, float)), (
            "目前只支持整数或浮点数作为指数"
        )

        out = Value(
            self.data ** other,
            (self,),
            f"**{other}"
        )

        def _backward():
            # dz/dx = n * x ** (n - 1)
            self.grad += (
                other
                * self.data ** (other - 1)
                * out.grad
            )

        out._backward = _backward

        return out

    def relu(self):
        """ReLU：z = max(0, x)"""
        out = Value(
            0.0 if self.data <= 0 else self.data,
            (self,),
            "ReLU"
        )

        def _backward():
            # x > 0时导数为1，否则为0
            local_grad = 1.0 if self.data > 0 else 0.0
            self.grad += local_grad * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """从当前节点开始执行完整的反向传播"""

        # 保存计算图的拓扑顺序
        topo = []

        # 防止同一个节点被重复访问
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                # 先访问当前节点依赖的所有父节点
                for child in node._prev:
                    build_topo(child)

                # 父节点处理完后，再加入当前节点
                topo.append(node)

        # 从最终输出节点开始构建拓扑顺序
        build_topo(self)

        # 最终结果对自己的梯度恒等于1
        self.grad = 1.0

        # 从最终节点向输入节点逆序传播
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"