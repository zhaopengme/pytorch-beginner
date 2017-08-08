import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


dtype = torch.FloatTensor

d = 2
h = 2

x = Variable(torch.ones(d,h))

