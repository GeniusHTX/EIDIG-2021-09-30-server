import pickle as pkl
import numpy as np


def watch_npy():
    data = np.load("C-a_ids_EIDIG_5_1.npy")
    return data


def watch_pkl():
    f = open("C-a_ids_EIDIG_5_1.pkl", 'rb')
    data = pkl.load(f)
    return data


if __name__ == "__main__":
    data = watch_pkl()
    print(data)
