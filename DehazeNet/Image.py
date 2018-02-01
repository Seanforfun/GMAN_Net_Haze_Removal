#  ====================================================
#   Filename: Image.py
#   Function: This file defines a Image class
#  ====================================================
import numpy as np


class Image(object):
    def __init__(self, path, key=None, image_matrix=None, image_index=None):
        self.path = path
        self.key = key
        self.image_matrix = image_matrix
        self.image_index = image_index


if __name__ == '__main__':
    i = Image("ASD", 12)
    print(i.path)
