#  ====================================================
#   Filename: Image.py
#   Function: This file defines a Image class
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Image(object):
    def __init__(self, path, key=None, image_tensor=None, image_index=None):
        self.path = path
        self.key = key
        self.image_tensor = image_tensor
        self.image_index = image_index


if __name__ == '__main__':
    pass
