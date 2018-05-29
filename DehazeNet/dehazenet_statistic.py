#  ====================================================
#   Filename: dehazenet_statistic.py
#   Function: This file used to evaluate the performance of  PSNR
#  according to depth of each pixel.
#  ====================================================
from PIL import Image as Image
import threadpool
import numpy as np
import os


GROUP_NUM = 10
CHANNEL_NUM = 3
# TODO Need to assign a directory
clear_dir = ""
hazy_dir = ""
depth_dir = ""


# TODO lz
def sta_image_input(clear_dir, result_dir, depth_dir):
    clear_file_list = os.listdir(clear_dir)
    result_file_list = os.listdir(result_dir)
    depth_file_list = os.listdir(depth_dir)

    return clear_file_list, result_file_list, depth_file_list

def sta_cal_single_image(clear, result, depth, psnr_map, group_id, divide):
    low_boundary = group_id * divide
    up_boundary = low_boundary + divide
    shape = np.shape(clear)
    H = shape[0]
    W = shape[1]
    depth_matting = np.zeros((H, W))
    count = 0
    for h in range(H):
        for w in range(W):
            single_pixel = depth[h][w]
            if(single_pixel >= low_boundary and single_pixel < up_boundary ):    # If depth is in the range
                count += 1
                depth_matting[h][w] = 1

    for i in range(CHANNEL_NUM):
        clear[:, :, i] = np.multiply(clear[:, :, i], depth_matting)
        result[:, :, i] = np.multiply(result[:, :, i], depth_matting)


def sta_read_image(clear_image_dir, result_image_dir, depth_map_dir):
    # read clear, haze, depth image into memory
    # Assert H and W are the same.
    # Return matrices for three images.
    clear_image = Image.open(clear_image_dir)
    clear = np.array(clear_image)
    result_image = Image.open(result_image_dir)
    result = np.array(result_image)
    depth_image = Image.open(depth_map_dir)
    depth = np.array(depth_image)
    return clear, result, depth


def sta_do_statistic(divide, thread_pool):
    # Read clear image, hazy image and their corresponding depth image.
    # Get the three corresponding matrices for a single image.
    clear_single_dir = ""
    result_single_dir = ""
    depth_single_dir = ""
    clear, result, depth = sta_read_image(clear_single_dir, result_single_dir, depth_single_dir)

    #  Traversal every pixel on the image and get a map for each group of the image.
    #  image in the same group = old image .* map
    task_list = []  # Create a list to save all tasks
    psnr_map = {}  # Map to write PSNR result.
    for i in range(GROUP_NUM):
        task_list.append(threadpool.makeRequests(sta_cal_single_image, [clear, result, depth, psnr_map, i, divide]))

    # Calculate PSNR for all of the group respectively and record them into a new file.
    pass


def main():
    # Group order: 0, 1, 2 ... GROUP_NUM-1
    divide = 1 / GROUP_NUM
    # Create a thread pool, # of thread = GROUP_NUM * 2.
    pool = threadpool.ThreadPool(GROUP_NUM * 2)
    # call dehazenet_input to read the images directory.
    clear_image_list, hazy_image_list, depth_image_list = sta_image_input(clear_dir, hazy_dir, depth_dir)
    #  Start doing statistic calculation
    sta_do_statistic(divide, pool)
    pass


if __name__ == '__main__':
    main()