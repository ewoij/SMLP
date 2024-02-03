import random


def relu(x):
    return max(0, x)


class Neuron:
    def __init__(self, nin: int) -> None:
        self.w = [random.uniform(0,1) for _ in range(nin)]
        self.x = [0] * nin
        self.b = random.uniform(0,1)
        self.out = 0.0
        self.reset_grad()

    def reset_grad(self) -> None:
        self.w_grad = [0.0] * len(self.w)
        self.x_grad = [0.0] * len(self.w)

    def __call__(self, x: list[float]) -> float:
        self.x = x
        self.out = sum(x * w for x, w in zip(x, self.w)) + self.b
        self.out = relu(self.out)
        return self.out

    def backward(self, grad: float) -> None:
        for i in range(len(self.w)):
            self.w_grad[i] += (self.x[i] if self.x[i] * self.w[i] > 0 else 0) * grad
            self.x_grad[i] += (self.w[i] if self.x[i] * self.w[i] > 0 else 0) * grad

    def __repr__(self, indent: int = 0) -> str:
        return '\n'.join([
            f"{' ' * indent}b={self.b}",
            '\n'.join(f"{' ' * indent}{' ' * 2}{i}: w={self.w[i]}, x={self.x[i]}, w_grad={self.w_grad[i]}, x_grad={self.x_grad[i]}" for i in range(len(self.w))),
        ])



class MLP:
    def __init__(self, shape: list[int]) -> None:
        assert len(shape) >= 2, "must have at leat 1 layer of neurons"
        self.layers = [
            [Neuron(shape[i-1]) for _ in range(shape[i])] 
            for i in range(1, len(shape))
        ]

    def __call__(self, x: list[float]) -> list[float]:
        assert len(x) == len(self.layers[0][0].w), "incorrect number of inputs"
        out = x
        for neurons in self.layers:
            out = [neuron(out) for neuron in neurons]
        return out

    def backward(self, expected: list[float]) -> None:
        assert len(expected) == len(self.layers[-1]), "must have same dimension as the last layer"
        actual = [neuron.out for neuron in self.layers[-1]]
        # actual > expected: following gradient will increase the loss
        # actual < expected: following gradient will decrease the loss, therefore reverting
        grad = [1 if actual[i] > expected[i] else -1 for i in range(len(self.layers[-1]))]
        for i in range(len(grad)):
            self.layers[-1][i].backward(grad[i])
        for li in reversed(range(1, len(self.layers))):
            for neuron in self.layers[li]:
                for i in range(len(neuron.w)):
                    self.layers[li-1][i].backward(neuron.x_grad[i])

    def descent(self, step: float = 0.1**3) -> None:
        for li in range(len(self.layers)):
            for ni in range(len(self.layers[li])):
                for wi in range(len(self.layers[li][ni].w)):
                    n = self.layers[li][ni]
                    w, g = n.w, n.w_grad
                    w[wi] += w[wi] * -g[wi] * step

    def reset_grad(self) -> None:
        for li in range(len(self.layers)):
            for ni in range(len(self.layers[li])):
                self.layers[li][ni].reset_grad()

    def __repr__(self) -> str:
        f_list = lambda l: '[\n' + '\n'.join([v.__repr__(indent=2) for v in l]) + '\n]'
        return '\n\n'.join(f'layer={i}, neurons={f_list(self.layers[i])}' for i in range(len(self.layers)))


m = MLP([2,10,10,2])

D = [
    [[0, 0], [1, 1]],
    [[0, 1], [1, 0]],
    [[1, 0], [0, 1]],
    [[1, 1], [0, 0]],
]

i = 0
while True:
    m.reset_grad()

    for in_, exp in D:
        out = m(in_)
        m.backward(out)

    m.descent(0.1**4)

    if i % 100 == 0:
        for in_, exp in D:
            out = m(in_)
            loss = sum((exp - out)**2 for exp, out in zip(exp, out))
            print(in_, out, loss)

        input()

    i+=1

m = MLP([1,1,1])

h = 0.1 ** 6

m.layers[0][0].w[0] = 2
m.layers[0][0].b = 0
m.layers[1][0].w[0] = 3
m.layers[1][0].b = 0

[a] = m([1])
m.layers[0][0].w[0] += h
[b] = m([1])
m.layers[0][0].w[0] -= h # revert
print('n0', (b - a) / h)

[a] = m([1])
m.layers[1][0].w[0] += h
[b] = m([1])
m.layers[1][0].w[0] -= h # revert
print('n1', (b - a) / h)
