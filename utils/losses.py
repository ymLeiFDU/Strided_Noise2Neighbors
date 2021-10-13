# coding:'utf-8'
from __future__ import division

import torch
import torch.nn as nn
import json
import os
import sys
import numpy as np
import torch.nn.functional as F

from math import exp
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
from shutil import copyfile


class MSELoss(_Loss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        loss = torch.mean((pred - target)**2)
        return loss



















