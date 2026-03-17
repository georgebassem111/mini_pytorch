import numpy as np

def sum_to_shape(grad, shape):
    # remove extra dimensions
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # sum along broadcasted axes
    for i in reversed(range(len(shape))):
        if shape[i] == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # -------- basic ops --------

    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other) if requires_grad else (), "+", requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += sum_to_shape(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += sum_to_shape(out.grad, other.data.shape)

        if requires_grad:
            out._backward = _backward

        return out

    def __mul__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other) if requires_grad else (), "*", requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += sum_to_shape(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += sum_to_shape(self.data * out.grad, other.data.shape)

        if requires_grad:
            out._backward = _backward

        return out


    def __matmul__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, (self, other) if requires_grad else (), "matmul", requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                self_grad = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                self.grad += sum_to_shape(self_grad, self.data.shape)
            if other.requires_grad:
                other_grad = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                other.grad += sum_to_shape(other_grad, other.data.shape)

        if requires_grad:
            out._backward = _backward

        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        out = Tensor(self.data.reshape(shape), (self,) if self.requires_grad else (), "reshape", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        if self.requires_grad:
            out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(np.transpose(self.data, axes), (self,) if self.requires_grad else (), "transpose", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad += np.transpose(out.grad)
                else:
                    inverse_axes = np.argsort(axes)
                    self.grad += np.transpose(out.grad, inverse_axes)

        if self.requires_grad:
            out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,) if self.requires_grad else (), "slice", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad[idx] += out.grad

        if self.requires_grad:
            out._backward = _backward
        return out

    def masked_fill(self, mask, value):
        mask = np.asarray(mask, dtype=bool)
        out = Tensor(np.where(mask, value, self.data), (self,) if self.requires_grad else (), "masked_fill", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += np.where(mask, 0.0, out.grad)

        if self.requires_grad:
            out._backward = _backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,) if self.requires_grad else (), "tanh", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 - out.data**2) * out.grad

        if self.requires_grad:
            out._backward = _backward
        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,) if self.requires_grad else (), "sigmoid", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.data * (1 - out.data) * out.grad

        if self.requires_grad:
            out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,) if self.requires_grad else (), "relu", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        if self.requires_grad:
            out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(probs, (self,) if self.requires_grad else (), "softmax", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
                self.grad += probs * (out.grad - grad_dot)

        if self.requires_grad:
            out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,) if self.requires_grad else (), "sum", requires_grad=self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return

            grad = out.grad
            if axis is None:
                self.grad += np.ones_like(self.data) * grad
                return

            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)

            if not keepdims:
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, axis=ax)

            self.grad += np.ones_like(self.data) * grad

        if self.requires_grad:
            out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,) if self.requires_grad else (), "mean", requires_grad=self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return

            grad = out.grad
            if axis is None:
                self.grad += np.ones_like(self.data) * grad / self.data.size
                return

            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
            count = np.prod([self.data.shape[ax] for ax in axes])

            if not keepdims:
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, axis=ax)

            self.grad += np.ones_like(self.data) * grad / count

        if self.requires_grad:
            out._backward = _backward
        return out


    def __neg__(self): 
        return self * -1
    def __radd__(self, other): 
        return self + other
    def __sub__(self, other): 
        return self + (-other)
    def __rsub__(self, other): 
        return other + (-self)
    def __rmul__(self, other): 
        return self * other
    def __truediv__(self, other): 
        return self * other**-1
    def __rtruediv__(self, other): 
        return other * self**-1

    def log(self):
        out = Tensor(np.log(self.data), (self,) if self.requires_grad else (), "log", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad

        if self.requires_grad:
            out._backward = _backward

        return out
        
    def exp(self):
        out = Tensor(np.exp(self.data), (self,) if self.requires_grad else (), "exp", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad

        if self.requires_grad:
            out._backward = _backward

        return out

    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,) if self.requires_grad else (), f"pow{power}", requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += power * (self.data ** (power - 1)) * out.grad

        if self.requires_grad:
            out._backward = _backward
        return out
    
    # -------- backward -------- DFS through the computational graph (topological sort) --------

    def backward(self):

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
