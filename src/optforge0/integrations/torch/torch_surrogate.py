import torch

class TorchRegressor:
    def __init__(self, model: torch.nn.Module, loss_fn, optimizer_lambda, nsteps, minimizer_lambda, min_steps, device = torch.device('cuda')):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer_lambda = optimizer_lambda
        self.minimizer_lambda = minimizer_lambda
        self.nsteps = nsteps
        self.min_steps = min_steps
        self.device = device

        self.last_x = None

    def fit(self, x:torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

        opt = self.optimizer_lambda(self.model.parameters())
        for i in range(self.nsteps):
            def closure(backward = True):
                yhat = self.model(x)[:, 0]
                loss = self.loss_fn(y, yhat)
                if backward:
                    opt.zero_grad()
                    loss.backward()
                return loss
            opt.step(closure)

        if self.last_x is None:
            self.last_x = x[torch.argmin(y)].unsqueeze(0)

    def minimize(self):
        x:torch.Tensor = self.last_x.clone().requires_grad_(True) # type:ignore
        opt = self.minimizer_lambda([x])

        for i in range(self.nsteps):
            def closure(backward = True):
                loss = self.model(x)
                if backward:
                    opt.zero_grad()
                    loss.backward()
                return loss
            opt.step(closure)

        self.last_x = x.detach()
        return self.last_x