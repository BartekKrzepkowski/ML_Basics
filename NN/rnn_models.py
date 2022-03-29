import torch
import torch.nn as nn


class RNNElman(nn.Module):
    def __init__(self, input_size, hidden_size, act1, act2):
        super().__init__()
        self.Wih = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bih = nn.Parameter(torch.Tensor(hidden_size))

        self.Why = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bhy = nn.Parameter(torch.Tensor(hidden_size))

        self.act1 = act1
        self.act2 = act2

    def forward(self, x, h):
        h = self.act1(x @ self.Wih + h @ self.W_h2 + self.bih)
        y = self.act2(h @ self.Why + self.bhy)
        return y, h


class RNNJordan(nn.Module):
    def __init__(self, input_size, hidden_size, act1, act2):
        super().__init__()
        self.Wih = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bih = nn.Parameter(torch.Tensor(hidden_size))

        self.Why = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bhy = nn.Parameter(torch.Tensor(hidden_size))

        self.act1 = act1
        self.act2 = act2

    def forward(self, x, y):
        h = self.act1(x @ self.Wih + y @ self.Whh + self.bih)
        y = self.act2(h @ self.Why + self.bhy)
        return y, h


class RNNCellPytorch(nn.Module):
    def __init__(self, input_size, hidden_size, act):
        super().__init__()
        self.Wih = torch.nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whh = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bih = nn.Parameter(torch.Tensor(hidden_size))
        self.bhh = nn.Parameter(torch.Tensor(hidden_size))

        self.act = act

    def forward(self, x, h):
        h = self.act(x @ self.Wih + self.bih + h @ self.Whh + self.bhh)
        return h


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, act1=torch.sigmoid, act2=torch.tanh, act3=torch.tanh):
        super().__init__()
        self.Wif = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhf = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))

        self.Wii = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhi = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))

        self.Wio = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uho = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))

        self.Wig = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhg = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bg = nn.Parameter(torch.Tensor(hidden_size))

        self.act1 = act1
        self.act2 = act2
        self.act3 = act3

    def forward(self, x, s):
        (h, c) = s
        f = self.act1(x @ self.Wif + h @ self.Uhf + self.bf)
        i = self.act1(x @ self.Wii + h @ self.Uhi + self.bi)
        o = self.act1(x @ self.Wio + h @ self.Uho + self.bo)
        g = self.act2(x @ self.Wig + h @ self.Uhg + self.bg)
        c = f * c + i * g
        h = o * self.act3(c)
        return o, (h, c)


class LSTMPipholeCell(nn.Module):
    def __init__(self, input_size, hidden_size, act1=torch.sigmoid, act2=torch.tanh, act3=lambda x: x):
        super().__init__()
        self.Wif = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhf = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))

        self.Wii = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhi = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))

        self.Wio = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uho = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))

        self.Wig = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.bg = nn.Parameter(torch.Tensor(hidden_size))

        self.act1 = act1
        self.act2 = act2
        self.act3 = act3

    def forward(self, x, s):
        (h, c) = s
        f = self.act1(x @ self.Wif + c @ self.Uhf + self.bf)
        i = self.act1(x @ self.Wii + c @ self.Uhi + self.bi)
        o = self.act1(x @ self.Wio + c @ self.Uho + self.bo)
        g = self.act2(x @ self.Wig + self.bg)
        c = f * c + i * g
        h = o * self.act3(c)
        return o, (h, c)


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, act1=torch.sigmoid, act2=torch.tanh):
        super().__init__()
        self.Wiz = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhz = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bz = nn.Parameter(torch.Tensor(hidden_size))

        self.Wir = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhr = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.br = nn.Parameter(torch.Tensor(hidden_size))

        self.Win = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhn = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bn = nn.Parameter(torch.Tensor(hidden_size))

        self.act1 = act1
        self.act2 = act2

    def forward(self, x, h):
        z = self.act1(x @ self.Wiz + h @ self.Uhz + self.bz)
        r = self.act1(x @ self.Wir + h @ self.Uhr + self.br)
        n = self.act1(x @ self.Win + (r * h) @ self.Uhn + self.bn)
        h = (1 - z) * h + z * n
        return h


class MGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, act1=torch.sigmoid, act2=torch.tanh):
        super().__init__()
        self.Wiz = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhz = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bz = nn.Parameter(torch.Tensor(hidden_size))

        self.Win = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Uhn = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bn = nn.Parameter(torch.Tensor(hidden_size))

        self.act1 = act1
        self.act2 = act2

    def forward(self, x, h):
        z = self.act1(x @ self.Wiz + h @ self.Uhz + self.bz)
        n = self.act1(x @ self.Win + (z * h) @ self.Uhn + self.bn)
        h = (1 - z) * h + z * n
        return h


class CARUCell(nn.Module):
    def __init__(self, input_size, hidden_size, act1=torch.sigmoid, act2=torch.tanh):
        super().__init__()
        self.Wvx = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.bx = nn.Parameter(torch.Tensor(hidden_size))

        self.Whn = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bn = nn.Parameter(torch.Tensor(hidden_size))

        self.Wvz = torch.nn.Parameters(torch.Tensor(input_size, hidden_size))
        self.Whz = torch.nn.Parameters(torch.Tensor(hidden_size, hidden_size))
        self.bz = nn.Parameter(torch.Tensor(hidden_size))

        self.act1 = act1
        self.act2 = act2

    def forward(self, v, h):
        x = v @ self.Wvx + self.bx
        n = self.act2(h @ self.Whn + self.bn + x)
        z = self.act1(h @ self.Whz + v @ self.Wvz + self.bz)
        l = self.act1(x) * z
        h = (1 - l) * h + l * n
        return h