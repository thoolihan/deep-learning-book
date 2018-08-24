import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

PLAYERS = 3
GRID_SHAPE = [4, 5]
STATE_SHAPE = GRID_SHAPE + [3]

print("creating players")
pid = range(PLAYERS)
player_v = torch.tensor(pid)
print(player_v)

print("\ncreating grid")
grid_m = torch.randn(GRID_SHAPE)
print(grid_m)

print("\ncreating state")
states = np.zeros(STATE_SHAPE)
states_t = torch.tensor(states)
print(states_t)


print("\nexamining state")
print(states_t[0, 0, 0])
print(states_t[0, 0, 0].item())

print("\ndouble grid difficulty")
grid_m = torch.mul(grid_m, 2)
print(grid_m)

print("\nexamine reshape grid")
grid_warped = grid_m.view(2, -1)
print(grid_warped)

print("\nexamine and use gradient function")
grid_log = torch.abs(grid_m)
grid_log.requires_grad_()
grid_log = torch.log(grid_log)
print(grid_log)
print(grid_log.grad_fn)
s = grid_log.sum()
s.backward()
print(grid_log.grad)