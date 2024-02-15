from abc import ABC
import random
import math
from typing import Type


class ActivationFunc(ABC):
    def __call__(self, x) -> float:
        self.x = x
        return 0

    @property
    def grad(self) -> float:
        raise NotImplemented()


class Relu(ActivationFunc):
    name = "Relu"

    def __call__(self, x) -> float:
        super().__call__(x)
        return max(0, x)

    @property
    def grad(self) -> float:
        return 1 if self.x > 0 else 0


class Tanh(ActivationFunc):
    name = "Tanh"

    def __call__(self, x) -> float:
        super().__call__(x)
        return math.tanh(x)

    @property
    def grad(self) -> float:
        return 1 - math.tanh(self.x)**2


class MeanSquareLoss:
    def __call__(self, x: list[float], e: list[float]) -> float:
        assert len(x) == len(e)
        self.x, self.e = x, e
        return sum((x[i] - e[i]) ** 2 for i in range(len(x))) / len(x)

    @property
    def grad(self) -> list[float]:
        x, e = self.x, self.e
        return [2 * (x[i] - e[i]) / len(x) for i in range(len(x))]


class Neuron:
    def __init__(self, nin: int, activation_func: ActivationFunc) -> None:
        self.w = [random.uniform(-1,1) for _ in range(nin)]
        self.x = [0] * nin
        self.b = random.uniform(-1,1)
        self.out = 0.0
        self.reset_grad()
        self.activation_func = activation_func

    def reset_grad(self) -> None:
        self.w_grad = [0.0] * len(self.w)
        self.w_grad_comp = [None] * len(self.w)
        self.x_grad = [0.0] * len(self.w)
        self.b_grad = 0.0
        self.b_grad_comp = 0.0

    def __call__(self, x: list[float]) -> float:
        self.x = x
        self.out = sum(x * w for x, w in zip(x, self.w)) + self.b
        self.out = self.activation_func(self.out)
        return self.out

    def backward(self, grad: float) -> None:
        self.b_grad += 1 * self.activation_func.grad * grad
        for i in range(len(self.w)):
            self.w_grad[i] += self.x[i] * self.activation_func.grad * grad
            self.x_grad[i] += self.w[i] * self.activation_func.grad * grad

    def __repr__(self, indent: int = 0) -> str:
        return '\n'.join([
            f"{' ' * indent}b={self.b}, b_grad={self.b_grad}, b_grad_comp={self.b_grad_comp}, out={self.out}, activation_func={self.activation_func.name}",
            '\n'.join(f"{' ' * indent}{' ' * 2}{i}: w={self.w[i]}, x={self.x[i]}, w_grad={self.w_grad[i]}, w_grad_comp={self.w_grad_comp[i]}, x_grad={self.x_grad[i]}" for i in range(len(self.w))),
        ])



class MLP:
    def __init__(self, nin: int, layers: list[list[Type[ActivationFunc]]]) -> None:
        assert len(layers) >= 1, "must have at least one layer"

        self.layers = []
        for layer in layers:
            self.layers.append([Neuron(nin, act_func_cls()) for act_func_cls in layer])
            nin = len(layer)

    def __call__(self, x: list[float]) -> list[float]:
        assert len(x) == len(self.layers[0][0].w), "incorrect number of inputs"
        out = x
        for neurons in self.layers:
            out = [neuron(out) for neuron in neurons]
        return out

    def backward(self, grad: list[float]) -> None:
        assert len(grad) == len(self.layers[-1]), "must have same dimension as the last layer"
        for i in range(len(grad)):
            self.layers[-1][i].backward(grad[i])
        for li in reversed(range(1, len(self.layers))):
            for neuron in self.layers[li]:
                for i in range(len(neuron.w)):
                    self.layers[li-1][i].backward(neuron.x_grad[i])

    def descent(self, step: float = 0.1**3) -> None:
        for li in range(len(self.layers)):
            for ni in range(len(self.layers[li])):
                n = self.layers[li][ni]
                n.b += -n.b_grad * step
                for wi in range(len(n.w)):
                    w, g = n.w, n.w_grad
                    w[wi] += -g[wi] * step

    def reset_grad(self) -> None:
        for li in range(len(self.layers)):
            for ni in range(len(self.layers[li])):
                self.layers[li][ni].reset_grad()

    def compute_grad_forward_pass(self, x: list[float]):
        h = 0.1**6
        for li in range(len(self.layers)):
            for ni in range(len(self.layers[li])):
                for wi in range(len(self.layers[li][ni].w)):
                    a = self(x)
                    self.layers[li][ni].w[wi] += h
                    b = self(x)
                    self.layers[li][ni].w[wi] -= h
                    self.layers[li][ni].w_grad_comp[wi] = [(b[i] - a[i]) / h for i in range(len(a))]

                    self.layers[li][ni].b += h
                    b = self(x)
                    self.layers[li][ni].b -= h
                    self.layers[li][ni].b_grad_comp = [(b[i] - a[i]) / h for i in range(len(a))]

    def __repr__(self) -> str:
        f_list = lambda l: '[\n' + '\n'.join([v.__repr__(indent=2) for v in l]) + '\n]'
        return '\n\n'.join(f'layer={i}, neurons={f_list(self.layers[i])}' for i in range(len(self.layers)))


def shuffle_data(D, E):
    combined = list(zip(D, E))
    random.shuffle(combined)
    D[:], E[:] = zip(*combined)
    return D, E


def random_sample_data(D, E, n):
    D, E = shuffle_data(D, E)
    return D[:n], E[:n]


def train(m: MLP, X: list[list[float]], Y: list[list[float]], num_epochs: int = 10**3, descent_step: float = 0.1 ** 3) -> None:
    loss = MeanSquareLoss()
    for epoch in range(num_epochs):
        X, Y = shuffle_data(X, Y)
        loss_sum = 0
        for i in range(len(X)):
            m.reset_grad()
            output = m(X[i])
            l = loss(output, Y[i])
            loss_sum += l
            m.backward(loss.grad)
            m.descent(step=descent_step)
        print(f"{epoch:6}: {loss_sum/len(X):<10}")
