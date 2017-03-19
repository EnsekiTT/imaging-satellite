# coding: utf-8
from os.path import join, relpath
import glob
import csv
import os
from PIL import Image, ImageFile
import numpy as np
import psycopg2
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
    with open('som.pickle', mode='rb') as f:
        som = pickle.load(f)
    with open(os.path.join(os.getcwd(), "static", "erodata.tsv"),'w') as df:
        fieldnames = ['x', 'y', 'uri']
        writer = csv.DictWriter(df,fieldnames=fieldnames, delimiter = '\t')
        writer.writeheader()
        for i, _file in enumerate(img_list):
            print(_file)
            try:
                np_img, raw = read_img(_file)
                if np_img.shape != (1, 9216):
                    continue
                x, y = som._BMU(np_img)
                writer.writerow({'x': x, 'y': y, 'uri': _file})
            except OSError as e:
                print(e.strerror)
                continue
            #if i < 100:
            #    continue

if __name__ == "__main__":
    main()
