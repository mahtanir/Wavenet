
#borrowed from https://medium.com/@kion.kim/wavenet-a-network-good-to-know-7caaae735435

import numpy as np
from torch import nn
import torch.nn.functional as F
import torch


def encode_mu_law(x, mu=256):
    mu = mu-1
    fx = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu) #1
    return np.floor((fx+1)/2*mu+0.5).astype(np.long) #2

def decode_mu_law(y, mu=256):
    mu = mu-1
    fx = (y-0.5)/mu*2-1 #reverse of #1
    x = np.sign(fx)/mu*((1+mu)**np.abs(fx)-1) 
    return x  

def one_hot_utils(x, kernel_size):
    x = torch.tensor(np.array(x))
    one_hot = F.one_hot(x, num_classes=256) 
    tf_shape = (1, -1, 256) #so rows actually are the points! I THINK! but then the way the conv channel works is weird... But image also shows like this 
    one_hot = torch.reshape(one_hot, tf_shape)
    return one_hot