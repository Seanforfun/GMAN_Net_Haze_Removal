import numpy as np


class Image(object):
    def __init__(self, path, key=None, image_matrix=None):
        self.path = path
        self.key = key
        self.image_matrix = image_matrix


if __name__ == '__main__':
    i = Image("ASD", 12)
    print(i.path)
