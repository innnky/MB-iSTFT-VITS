filelist = "filelists/train.txt"
import librosa
import pyworld
import numpy as np
# import utils


fs = 22050
hop = 256


def compute_f0(path):
    x, sr = librosa.load(path, sr=fs)
    assert sr == fs
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=800,
        frame_period=1000 * hop / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, fs)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return f0

i = 0

with open(filelist) as f:
    for line in f.readlines():
        wavpath = line.split("|")[0]
        f0 = compute_f0(wavpath)
        print(wavpath, i)
        i += 1
        np.save(wavpath+".f0.npy", f0)


