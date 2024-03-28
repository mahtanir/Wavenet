import numpy as np
from utils import * 
from dataloader import * 
from IPython.display import Audio

def play_audio():
    arr = np.load('./wav.npy')
    fr, data = load_music('data_parametric-2')
    Audio(arr, fr)

play_audio()
