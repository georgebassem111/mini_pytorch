import numpy as np
from micrograd.engine import Tensor


class Module:
    def __init__(self):
        self.requires_grad = True

    def children(self):
        modules = []

        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                modules.append(attr)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        modules.append(item)

        return modules

    def parameters(self):

        params = []

        for attr in self.__dict__.values():

            if isinstance(attr, Tensor):
                params.append(attr)

            elif isinstance(attr, Module):
                params += attr.parameters()

            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        params += item.parameters()

        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def train(self):
        self.requires_grad = True
        self.check_requires_grad()
        for child in self.children():
            child.train()
        return self

    def eval(self):
        self.requires_grad = False
        self.check_requires_grad()
        for child in self.children():
            child.eval()
        return self

    def check_requires_grad(self):
        for p in self.parameters():
            p.requires_grad = self.requires_grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.W = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2/in_features), requires_grad=self.requires_grad)
        if bias:
            self.b = Tensor(np.random.randn(1, out_features) * 0.01, requires_grad=self.requires_grad)

    def forward(self, x):
        return x @ self.W + (self.b if hasattr(self, 'b') else 0)


class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Tanh(Module):
    def forward(self, x):
        return x.tanh()

class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x.softmax(axis=self.axis)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim) * 0.02, requires_grad=self.requires_grad,)

    def forward(self, x):
        idx = np.asarray(x, dtype=np.int64)
        out = Tensor(self.weight.data[idx], (self.weight,) if self.weight.requires_grad else (), "embedding", requires_grad=self.weight.requires_grad,)

        def _backward():
            if self.weight.requires_grad:
                np.add.at(self.weight.grad, idx, out.grad)

        if out.requires_grad:
            out._backward = _backward

        return out


class RMSNorm(Module):
    def __init__(self, dim=None, eps=1e-5, affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine and dim is not None

        if self.affine:
            self.weight = Tensor(np.ones((dim,), dtype=float), requires_grad=self.requires_grad)

    def forward(self, x):
        x_sq_mean = (x * x).mean(axis=-1, keepdims=True)
        normalized = x * ((x_sq_mean + self.eps) ** -0.5)

        if self.affine:
            return normalized * self.weight
        return normalized


class CrossEntropyLoss(Module):
    def forward(self, pred, target):
        m = pred.data.shape[0]
        target_idx = np.asarray(target.data, dtype=np.int64).reshape(-1)

        shifted = pred.data - np.max(pred.data, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        log_likelihood = -np.log(probs[np.arange(m), target_idx])
        loss = np.sum(log_likelihood) / m
        out = Tensor(loss, (pred,), "cross_entropy", requires_grad=pred.requires_grad and self.requires_grad)

        def _backward():
            grad = probs.copy()
            grad[np.arange(m), target_idx] -= 1.0
            grad /= m
            pred.grad += grad * out.grad

        if out.requires_grad:
            out._backward = _backward

        return out
