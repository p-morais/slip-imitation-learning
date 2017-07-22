import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, obs_dim, action_dim,  hidden_dims=(128, 128),
                 nonlin=F.relu, optimizer=optim.Adam):

        super(MLP, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers += [nn.Linear(obs_dim, hidden_dims[0])]
        for dim in hidden_dims:
            self.hidden_layers += [nn.Linear(dim, dim)]

        self.out = nn.Linear(hidden_dims[-1], action_dim)

        self.nonlin = nonlin

        print(self.nonlin)

        self.optimizer = optimizer(self.parameters(),
                                   lr=0.003,
                                   weight_decay=0.0)

        print(self)

    def forward(self, x):
        for l in self.hidden_layers:
            x = self.nonlin(l(x))

        x = self.out(x)

        return x

    def loss(self, X, y):
        output = self.forward(X)
        loss = nn.MSELoss()(output, y)
        return loss

    def fit(self, X, y):
        self.optimizer.zero_grad()
        output = self.forward(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()  # accumulate gradients
        self.optimizer.step()  # update parameters

        return loss.data
