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
        self.w_grad_comp = [None] * len(self.w)
        self.x_grad = [0.0] * len(self.w)
        self.b_grad = 0.0
        self.b_grad_comp = 0.0

    def __call__(self, x: list[float]) -> float:
        self.x = x
        self.out = sum(x * w for x, w in zip(x, self.w)) + self.b
        self.out = relu(self.out)
        return self.out

    def backward(self, grad: float) -> None:
        if self.out > 0:
            self.b_grad += 1 * grad 
            for i in range(len(self.w)):
                self.w_grad[i] += self.x[i] * grad
                self.x_grad[i] += self.w[i] * grad

    def __repr__(self, indent: int = 0) -> str:
        return '\n'.join([
            f"{' ' * indent}b={self.b}, b_grad={self.b_grad}, b_grad_comp={self.b_grad_comp}, out={self.out}",
            '\n'.join(f"{' ' * indent}{' ' * 2}{i}: w={self.w[i]}, x={self.x[i]}, w_grad={self.w_grad[i]}, w_grad_comp={self.w_grad_comp[i]}, x_grad={self.x_grad[i]}" for i in range(len(self.w))),
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
                n = self.layers[li][ni]
                n.b += n.b * -n.b_grad * step
                for wi in range(len(n.w)):
                    w, g = n.w, n.w_grad
                    w[wi] += w[wi] * -g[wi] * step

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

# end


def shuffle_data(D, E):
    combined = list(zip(D, E))
    random.shuffle(combined)
    D[:], E[:] = zip(*combined)
    return D, E


def random_sample_data(D, E, n):
    D, E = shuffle_data(D, E)
    return D[:n], E[:n]


def mean_square_loss(O: list[float], E: list[float]) -> float:
    return sum((O[i] - E[i])**2 for i in range(len(O))) / len(O)


m = MLP([2,2,2,1])
D = [
    [0, 0],
    [1, 1],
]
E = [
    [0],
    [1],
]

num_epochs = 10000
for epoch in range(num_epochs):
    D, E = shuffle_data(D, E)
    for i in range(len(D)):
        output = m(D[i])
        print(output)
        m.reset_grad()
        m.backward(E[i])
        m.descent()



m = MLP([2] + [10] * 2 + [2])
D = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
E = [
    [2, 2],
    [2, 1],
    [1, 2],
    [1, 1],
]
D, E = random_sample_data(D, E, 3) # strugles from 3

num_epochs = 10000
for epoch in range(num_epochs):
    O = [None] * len(D)
    D, E = shuffle_data(D, E)
    for i in range(len(D)):
        O[i] = m(D[i])
        m.reset_grad()
        m.backward(E[i])
        m.descent(step=0.1**3)
    if epoch % 100 == 0:
        loss = sum(mean_square_loss(O[i], E[i]) for i in range(len(O))) / len(O)
        print(loss)


# works pretty well for that
m = MLP([2] + [10] * 5 + [1])
D = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
E = [
    [1],
    [2],
    [3],
    [4],
]


num_epochs = 10000
for epoch in range(num_epochs):
    O = [None] * len(D)
    D, E = shuffle_data(D, E)
    for i in range(len(D)):
        O[i] = m(D[i])
        m.reset_grad()
        m.backward(E[i])
        m.descent(step=0.1**3)
    if epoch % 100 == 0:
        loss = sum(mean_square_loss(O[i], E[i]) for i in range(len(O))) / len(O)
        print(loss)
