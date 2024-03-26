import librosa
import numpy as np
import matplotlib.pyplot as plt

class music_data():

    def __init__(self):
        self.audio = None 
    
    def sample_audio():
        filename = librosa.util.example_audio_file()
        audio, _ = librosa.load(filename, 11025, mono=True, duration = 0.1)
        audio = audio.reshape(-1, 1)
        return audio

    def mu_law_encode(audio, quantization_channels):
        mu = tf.to_float(quantization_channels - 1)
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)

