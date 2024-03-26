import os 
from scipy.io import wavfile
import numpy as np 
from utils import * 

def load_music(music_name):
    fs, data = wavfile.read(os.path.join('data', music_name + '.wav'))
    # print(fs, data)
    return fs, data

def load_music_test():
    music_name = 'data_parametric-2'
    fs, data = load_music(music_name)
    print(fs, data, data.shape)

def data_generation(data, frame_rate, seq_size, mu, ctx):
    max_val = max(abs(min(data)), max(data))
    data = data / max_val
    while True: #forever? 
        sequence_sample_start = np.random.randint(0, data.shape[0] - seq_size)
        subsequence = data[sequence_sample_start: sequence_sample_start + seq_size]
        condensed_subsequence = encode_mu_law(subsequence, mu)
        yield condensed_subsequence #yield returns a generator object that is an iterable that can be iterated on (i.e with for loop) only once
    # preserves memory since it doesn't store it in memory vs other iterables like arrays or lists (function continues where left off
    # ) Returns one value at a time, as long as it knows the next I think it's ok. 

def data_generation_sample(data, frame_rate, seq_size, mu, ctx):
    #same logic as before but now we only return one, not a generator. 
    max_val = max(max(data), abs(min(data)))
    data = data / max_val 
    start = np.random.randint(0, data.shape[0] - seq_size)
    subset = data[start: start+ seq_size]
    return encode_mu_law(subset, mu)





load_music_test()