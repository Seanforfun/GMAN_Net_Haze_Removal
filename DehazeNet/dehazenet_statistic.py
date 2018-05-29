#  ====================================================
#   Filename: dehazenet_statistic.py
#   Function: This file used to evaluate the performance of  PSNR
#  according to depth of each pixel.
#  ====================================================
from PIL import Image as Image
from threadpool import *


GROUP_NUM = 10


def sta_read_image(clear_image_dir, haze_image_dir, depth_map_dir):
    # read clear, haze, depth image into memory
    # Assert H and W are the same.
    # Return matrices for three images.
    clear_image = Image.open(clear_image_dir)
    hazy_image = Image.open(haze_image_dir)
    depth_image = Image.open(depth_map_dir)
    return clear_image, hazy_image, depth_image


def do_statistic(divide, thread_pool):
    # Read clear image, hazy image and their corresponding depth image.
    clear, hazy, depth = sta_read_image()

    #  Traversal every pixel on the image and get a map for each group of the image.
    #  image in the same group = old image .* map

    # Calculate PSNR for all of the group respectively and record them into a new file.
    pass


def main():
    # Group order: 0, 1, 2 ... GROUP_NUM-1
    divide = 1 / GROUP_NUM
    # Create a thread pool, # of thread = GROUP_NUM * 2.
    pool = ThreadPool(GROUP_NUM * 2)

    #  Start doing statistic calculation
    do_statistic(divide, pool)
    pass


if __name__ == '__main__':
    main()