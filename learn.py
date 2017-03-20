# coding: utf-8
from os.path import join, relpath
import glob
import csv
import os
from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt
import pickle
from som import SOM
ImageFile.LOAD_TRUNCATED_IMAGES = True
in_path = 'static/images'

def read_img(_file):
    img = Image.open(_file)
    mini_img = img.resize((64, 48))
    np_img = np.asarray(mini_img)
    height, width, color = np_img.shape
    return np_img.reshape(1, height*width*color), img

def main():
    img_list = glob.glob(join(in_path, '*.jpg'))
    vectors = np.empty((0,64*48*3), int)
    for i, _file in enumerate(img_list):
        print(_file)
        try:
            np_img, raw = read_img(_file)
        except OSError as e:
            print(e.strerror)
            continue
        vectors = np.append(vectors, np_img, axis=0)
        #if i > 100:
        #    break

    N = 200
    som = SOM(vectors, N=N, seed=10)

    # learn
    som.learn(vectors)
    with open('som.pickle', mode='wb') as f:
        pickle.dump(som, f)
    print(som.nodes.shape)


if __name__ == "__main__":
    main()
