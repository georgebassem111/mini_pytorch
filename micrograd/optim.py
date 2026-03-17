import numpy as np

class Gradient:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)

    def step(self): 
        for p in self.parameters:
            p.data -= self.lr * p.grad


class Adam:
    def __init__(self, parameters, lr=0.01, betas=(0.85, 0.99), eps=1e-8):
        self.parameters = parameters
        self.lr = lr

        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)

    def step(self): 
        for i, p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** (self.t + 1))
            v_hat = self.v[i] / (1 - self.beta2 ** (self.t + 1))
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
            
        self.t += 1


class Muon:
    def __init__(self, parameters, lr=0.001, beta=0.9, steps=5, eps=1e-07):
        self.parameters = parameters
        self.lr = lr

        self.beta = beta
        self.steps = steps
        self.eps = eps

        self.m = [np.zeros_like(p.data) for p in self.parameters]
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)

    def NewtonSchulz5(self, G):
        a, b, c = (3.4445, -4.7750, 2.0315)

        # normalize G
        X = G / (np.linalg.norm(G, 'fro') + self.eps)

        # quintic polynomial approximation
        for _ in range(self.steps):
            A = X @ X.T
            B = b*A + c*A @ A
            X = a * X + B @ X
        
        return X

    def step(self): 
        for i, p in enumerate(self.parameters):
            # momentum
            self.m[i] = self.beta * self.m[i] + (1 - self.beta) * p.grad

            # orthogonalized update
            update = self.NewtonSchulz5(self.m[i])

            p.data -= self.lr * update