#  ====================================================
#   Filename: dehazenet_statistic.py
#   Function: This file used to evaluate the performance of  PSNR
#  according to depth of each pixel.
#  ====================================================
from PIL import Image as Image
import threadpool
import numpy as np
import os
import threading
from queue import PriorityQueue


GROUP_NUM = 10
CHANNEL_NUM = 3
COMMON_INDEX_BIT = 8
CLEAR_DICTIONARY = {}
DEPTH_DICTIONARY = {}
q = PriorityQueue() # Priority queue used to save pixel psnr information in increasing order, need lock

# TODO Need to assign a directory
clear_dir = ""
result_dir = ""
depth_dir = ""

q_lock = threading.Lock()


class PixelResult(object):
    def __init__(self, depth, psnr):
        self.depth = depth
        self.psnr = psnr

    def __cmp__(self, other):
        # Override __cmp__ for taking advantage of priority queue
        return self.depth - other.depth


def sta_cal_psnr(im1, im2, area, count):
    '''
        assert pixel value range is 0-255 and type is uint8
    '''
    # TODO Add psnr calculation according to pixel number
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean() * area
    mse /= count
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

def sta_cal_psnr_pixel(pixel1, pixel2):
    psnr = 0
    # Calculate psnr for a pixel
    return psnr

# Read clear and depth images from dictionary and create two dictionary for the files for referencing
def sta_image_input(clear_dir, depth_dir):
    # First read the file names from clear directory
    clear_file_list = os.listdir(clear_dir)
    depth_file_list = os.listdir(depth_dir)
    # Construct a dictionary for clear images
    for clear_image_name in clear_file_list:
        file_path = os.path.join(clear_dir, clear_image_name)
        clear_index = clear_image_name[0:COMMON_INDEX_BIT]
        CLEAR_DICTIONARY[clear_index] = file_path

    # Construct a dictionary for depth_images
    for depth_image_name in depth_file_list:
        file_path = os.path.join(depth_dir, depth_image_name)
        depth_index = depth_image_name[0:COMMON_INDEX_BIT]
        DEPTH_DICTIONARY[depth_index] = file_path
    return


# Process single image using by depth area.
def sta_cal_single_image_by_area(clear, result, depth, psnr_map, group_id, divide):
    low_boundary = group_id * divide
    up_boundary = low_boundary + divide
    shape = np.shape(clear)
    H = shape[0]
    W = shape[1]
    area = H * W
    depth_matting = np.zeros((H, W))
    count = 0
    for h in range(H):
        for w in range(W):
            single_pixel = depth[h][w]
            if low_boundary <= single_pixel < up_boundary :    # If depth is in the range
                count += 1
                depth_matting[h][w] = 1
    temp_clear = np.zeros((H, W, 3))
    temp_result = np.zeros((H, W, 3))
    for i in range(CHANNEL_NUM):
        temp_clear[:, :, i] = np.multiply(clear[:, :, i], depth_matting)
        temp_result[:, :, i] = np.multiply(result[:, :, i], depth_matting)

    psnr = sta_cal_psnr(temp_clear, temp_result, area, count)
    q_lock.acquire()
    psnr_map[group_id] = psnr
    q_lock.lease()


def sta_cal_single_image_by_pixel(clear, result, depth, q, group_id, divide):
    low_boundary = group_id * divide
    up_boundary = low_boundary + divide
    shape = np.shape(clear)
    H = shape[0]
    W = shape[1]
    for h in range(H):
        for w in range(W):
            pixel_depth = depth[h][w]
            if low_boundary <= pixel_depth < up_boundary:
                # Calculate psnr for single pixel and save into priority queue
                pixel_psnr = sta_cal_psnr_pixel(clear[h, w, :], result[h, w, :])
                pixel_result = PixelResult(pixel_depth, pixel_psnr)
                q_lock.acquire()
                q.put(pixel_result)
                q_lock.lease()


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
    result_file_list = os.listdir(result_dir)
    # Read clear image, hazy image and their corresponding depth image.
    # Get the three corresponding matrices for a single image.
    for result_image_name in result_file_list:
        result_single_dir = os.path.join(result_dir, result_image_name)
        index = result_image_name[0:COMMON_INDEX_BIT]
        clear_single_dir = CLEAR_DICTIONARY[index]
        depth_single_dir = DEPTH_DICTIONARY[index]
        clear, result, depth = sta_read_image(clear_single_dir, result_single_dir, depth_single_dir)
        #  Traversal every pixel on the image and get a map for each group of the image.
        #  image in the same group = old image .* map
        # Calculate PSNR for all of the group respectively and record them into psnr_map.
        task_list = []  # Create a list to save all tasks
        psnr_map = {}  # Map to write PSNR result. Mutual information, need to add a lock.
        for i in range(GROUP_NUM):
            lst_vars = [clear, result, depth, psnr_map, i, divide]
            func_var = [(lst_vars, None)]
            task_list.append(threadpool.makeRequests(sta_cal_single_image_by_area, func_var))
        for requests in task_list:
            [thread_pool.putRequest(req) for req in requests]
        thread_pool.poll()

        # Write the result into specific file.


def main():
    # Group order: 0, 1, 2 ... GROUP_NUM-1
    divide = 1 / GROUP_NUM
    # Create a thread pool, # of thread = GROUP_NUM * 2.
    pool = threadpool.ThreadPool(GROUP_NUM * 2)
    # call dehazenet_input to read the images directory.
    sta_image_input(clear_dir, depth_dir)
    #  Start doing statistic calculation
    sta_do_statistic(divide, pool)


if __name__ == '__main__':
    main()
