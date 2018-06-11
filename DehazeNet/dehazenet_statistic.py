#  ====================================================
#   Filename: dehazenet_statistic.py
#   Function: This file used to evaluate the performance of  PSNR
#  according to transmission of each pixel.
#  ====================================================
from PIL import Image as Image
import threadpool
import numpy as np
import os
import threading
import queue
import pickle
import math
import time
import multiprocessing


GROUP_NUM = 10
CHANNEL_NUM = 3
CLEAR_INDEX_BIT = 4
TRANS_INDEX_BIT = 4
NEED_SERIALIZATION = True
CLEAR_DICTIONARY = {}
TRANSMISSION_DICTIONARY = {}
SERIALIZATION_FILE_NAME = './PQ.pkl'
START_CALCULATION = True
FINAL_RESULT_MAP = {}   # key is the lowest transmission in current group, value is average mse.
# q = PriorityQueue()  # Priority queue used to save pixel psnr information in increasing order, need lock

CLEAR_DIR = "./ClearImages/TestImages"
RESULT_DIR = "./ClearResultImages"
TRANSMISSION_DIR = "./ClearImages/TransImages"

q_lock = threading.Lock()
START_CONDITION = threading.Condition()


class StatisticProducer(threading.Thread):
    def __init__(self, task_queue, result_dir):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.result_dir = result_dir

    def run(self):
        result_file_list = os.listdir(self.result_dir)
        # Read clear image, result image and their corresponding transmission image.
        # Get the three corresponding matrices for a single image.
        for result_image_name in result_file_list:
            result_single_dir = os.path.join(RESULT_DIR, result_image_name)
            clear_index = result_image_name[0:CLEAR_INDEX_BIT]
            trans_index = result_image_name[0:TRANS_INDEX_BIT]
            if clear_index not in CLEAR_DICTIONARY:
                raise RuntimeError(result_image_name + ' cannot find corresponding clear image.')
            clear_single_dir = CLEAR_DICTIONARY[clear_index]
            if trans_index not in TRANSMISSION_DICTIONARY:
                raise RuntimeError(result_image_name + ' cannot find corresponding transmission image.')
            transmission_single_dir = TRANSMISSION_DICTIONARY[trans_index]
            clear, result, transmission = sta_read_image(clear_single_dir, result_single_dir, transmission_single_dir)
            current_task = ImageTask(clear, result, transmission)
            self.task_queue.put(current_task)
            if START_CONDITION.acquire():
                START_CONDITION.notifyAll()
            START_CONDITION.release()
        self.task_queue.put(None)
        print('StatisticProducer finish')


class StatisticConsumer(threading.Thread):
    def __init__(self, pq, task_queue):
        threading.Thread.__init__(self)
        self.pq = pq
        self.task_queue = task_queue

    def run(self):
        if START_CONDITION.acquire():
            START_CONDITION.wait()
        START_CONDITION.release()
        while True:
            task = self.task_queue.get()
            if task is None:
                self.task_queue.put(None)
                break
            else:
                # Put the results into the priority queue
                sta_cal_single_image_by_pixel(task.clear_arr, task.result_arr, task.trans_arr, self.pq)
                time.sleep(0.0001)  # Sleep for 1 millisecond
        print('StatisticConsumer finish')


class PixelResult(object):
    def __init__(self, transmission, mse):
        self.transmission = transmission
        self.mse = mse

    def __lt__(self, other):
        return self.transmission < other.transmission


class ImageTask:
    def __init__(self, clear_image_arr, result_image_arr, transmission_arr):
        self.clear_arr = clear_image_arr
        self.result_arr = result_image_arr
        self.trans_arr = transmission_arr


def sta_cal_psnr(im1, im2, area, count):
    '''
        assert pixel value range is 0-255 and type is uint8
    '''
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean() * area
    mse /= count
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def sta_cal_psnr_pixel(pixel1, pixel2):
    psnr = 0
    # Calculate psnr for a pixel
    return psnr


def sta_cal_mse_pixel(pixel1, pixel2):
    return ((pixel1.astype(np.float) - pixel2.astype(np.float)) ** 2).mean()
    # return pixel1.astype(np.float) - pixel2.astype(np.float)


# Read clear and transmission images from dictionary and create two dictionary for the files for referencing
def sta_image_input(clear_dir, transmission_dir):
    # First read the file names from clear directory
    clear_file_list = os.listdir(clear_dir)
    transmission_file_list = os.listdir(transmission_dir)
    # Construct a dictionary for clear images
    for clear_image_name in clear_file_list:
        file_path = os.path.join(clear_dir, clear_image_name)
        clear_index = clear_image_name[0:CLEAR_INDEX_BIT]
        CLEAR_DICTIONARY[clear_index] = file_path

    # Construct a dictionary for transmission_images
    for transmission_image_name in transmission_file_list:
        file_path = os.path.join(transmission_dir, transmission_image_name)
        transmission_index = transmission_image_name[0:TRANS_INDEX_BIT]
        TRANSMISSION_DICTIONARY[transmission_index] = file_path


# Process single image using by transmission area.
def sta_cal_single_image_by_area(clear, result, transmission, psnr_map, group_id, divide):
    low_boundary = group_id * divide
    up_boundary = low_boundary + divide
    shape = np.shape(clear)
    H = shape[0]
    W = shape[1]
    area = H * W
    transmission_matting = np.zeros((H, W))
    count = 0
    for h in range(H):
        for w in range(W):
            single_pixel = transmission[h][w]
            if low_boundary <= single_pixel < up_boundary :    # If transmission is in the range
                count += 1
                transmission_matting[h][w] = 1
    temp_clear = np.zeros((H, W, 3))
    temp_result = np.zeros((H, W, 3))
    for i in range(CHANNEL_NUM):
        temp_clear[:, :, i] = np.multiply(clear[:, :, i], transmission_matting)
        temp_result[:, :, i] = np.multiply(result[:, :, i], transmission_matting)

    psnr = sta_cal_psnr(temp_clear, temp_result, area, count)
    q_lock.acquire()
    psnr_map[group_id] = psnr
    q_lock.release()


def sta_cal_single_image_by_pixel(clear, result, transmission, q):
    shape = np.shape(clear)
    H = shape[0]
    W = shape[1]
    for h in range(H):
        for w in range(W):
            pixel_transmission = transmission[h][w]
            pixel_mse = sta_cal_mse_pixel(clear[h, w, :], result[h, w, :])
            pixel_result = PixelResult(pixel_transmission, pixel_mse)
            q.put(pixel_result)


def sta_read_image(clear_image_dir, result_image_dir, transmission_map_dir):
    # read clear, haze, transmission image into memory
    # Assert H and W are the same.
    # Return matrices for three images.
    clear_image = Image.open(clear_image_dir)
    clear = np.array(clear_image)
    result_image = Image.open(result_image_dir)
    result = np.array(result_image)
    # the Transmission map might save as .npy file
    if transmission_map_dir.endswith(".npy"):
        transmission = np.load(transmission_map_dir)
    else:
        transmission_image = Image.open(transmission_map_dir)
        transmission = np.array(transmission_image)
    return clear, result, transmission


def sta_group_count_average(low, group_length, result_list):
    up = low + group_length
    total_mse = 0.0
    t = result_list[low].transmission
    while low < up:
        single_pixel_result = result_list[low]
        total_mse += single_pixel_result.mse
        low += 1
    q_lock.acquire()
    FINAL_RESULT_MAP[t] = total_mse/group_length
    q_lock.release()


def sta_create_visual_result(result_list, pool):
    # result_list is a list used to save the PixelResults, which is sorted at the previous step.
    result_len = len(result_list)
    '''
        How are lists implemented?
    Python’s lists are really variable-length arrays, not Lisp-style linked lists. The implementation uses a contiguous array of references to other objects, and keeps a pointer to this array and the array’s length in a list head structure.
    This makes indexing a list a[i] an operation whose cost is independent of the size of the list or the value of the index.
    When items are appended or inserted, the array of references is resized. Some cleverness is applied to improve the performance of appending items repeatedly; when the array must be grown, some extra space is allocated so the next few times don’t require an actual resize.
    '''
    # Internally is an array saving pointer, like arrayList in Java.
    # Calculate upper and lower index boundary for each group in list.
    group_length = math.floor(result_len / GROUP_NUM)
    task_list = []
    for i in range(GROUP_NUM):
        low = group_length * i
        lst_vars = [low, group_length, result_list]
        func_var = [(lst_vars, None)]
        task_list.append(threadpool.makeRequests(sta_group_count_average, func_var))
    for requests in task_list:
        [pool.putRequest(req) for req in requests]
    pool.wait()


def main():
    # Group order: 0, 1, 2 ... GROUP_NUM-1
    sorted_pickle_list = []
    CPU_NUM = multiprocessing.cpu_count()
    if START_CALCULATION:
        # call dehazenet_input to read the images directory.
        sta_image_input(CLEAR_DIR, TRANSMISSION_DIR)
        q = queue.PriorityQueue()
        task_queue = queue.Queue()
        thread_list = []
        #  Start doing statistic calculation
        statistic_producer = StatisticProducer(task_queue, RESULT_DIR)
        statistic_producer.start()
        thread_list.append(statistic_producer)

        for producer_id in range(CPU_NUM):
            statistic_consumer = StatisticConsumer(q, task_queue)
            statistic_consumer.start()
            thread_list.append(statistic_consumer)

        for thread in thread_list:
            thread.join()
        print('Step 1 : Producer-Consumer model calculation finish, Start doing statistical calculation.')

        while not q.empty():
            sorted_pickle_list.append(q.get())
        del q
        print("Step 2 : Finish copying queue to the list.")
        # Serialization the priority queue to SERIALIZATION_FILE_NAME
        if NEED_SERIALIZATION:
            if os.path.exists(SERIALIZATION_FILE_NAME):
                os.remove(SERIALIZATION_FILE_NAME)
            with open(SERIALIZATION_FILE_NAME, 'wb') as f:
                pickle.dump(sorted_pickle_list, f)   # Dump the queue into file
            print("Step 2.1 : Finish dump list to file.")

    else:
        # Load the queue from file
        print("Step 1 : Start loading dump file.")
        if not os.path.exists(SERIALIZATION_FILE_NAME):
            raise RuntimeError("Serialization file does not exist!")
        else:
            with open(SERIALIZATION_FILE_NAME, 'rb') as f:
                sorted_pickle_list = pickle.load(f) # load priority queue from file
            print("Step 2 : Finish loading the list from .pkl file.")

    # Use the data from calculation or serialization file to create the statistical result
    # Create a thread pool, # of thread = GROUP_NUM * 2.
    pool = threadpool.ThreadPool(GROUP_NUM * 2)
    sta_create_visual_result(sorted_pickle_list, pool)
    print("Step 3 : Finish calculating visual result.")
    print(FINAL_RESULT_MAP)


if __name__ == '__main__':
    main()
