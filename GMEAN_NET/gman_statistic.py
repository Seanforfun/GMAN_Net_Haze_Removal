#  ====================================================
#   Filename: gman_statistic.py
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
import gman_transmission
import multiprocessing
import time
import gman_constant as constant


q_lock = threading.Lock()
START_CONDITION = threading.Condition()
RESULT_IMAGE_QUEUE = queue.Queue()


class StatisticProducer(threading.Thread):
    def __init__(self, task_queue, result_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            # Read clear image, result image and their corresponding transmission image.
            # Get the three corresponding matrices for a single image.
            result_image_name = self.result_queue.get()
            if result_image_name is None:
                RESULT_IMAGE_QUEUE.put(None)
                self.task_queue.put(None)
                break
            result_single_dir = os.path.join(constant. STATS_RESULT_DIR, result_image_name)
            clear_index = result_image_name[0:constant. STATS_CLEAR_INDEX_BIT]
            trans_index = result_image_name[0:constant. STATS_TRANS_INDEX_BIT]
            if clear_index not in constant. STATS_CLEAR_DICTIONARY:
                raise RuntimeError(result_image_name + ' cannot find corresponding clear image.')
            clear_single_dir = constant. STATS_CLEAR_DICTIONARY[clear_index]
            if trans_index not in constant. STATS_TRANSMISSION_DICTIONARY:
                raise RuntimeError(result_image_name + ' cannot find corresponding transmission image.')
            transmission_single_dir = constant. STATS_TRANSMISSION_DICTIONARY[trans_index]
            clear, result, transmission = sta_read_image(clear_single_dir, result_single_dir, transmission_single_dir)
            [_, filename] = os.path.split(transmission_single_dir)
            _, alpha, beta = gman_transmission.trans_get_alpha_beta(filename)
            current_task = ImageTask(clear, result, transmission, alpha, beta)
            self.task_queue.put(current_task)
        print('Statistic Producer finish')


class StatisticConsumer(threading.Thread):
    producer_end_number = 0

    def __init__(self, task_queue, producer_number, lock, bag):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.producer_number = producer_number
        self.lock = lock
        self.bag = bag

    def run(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                self.lock.acquire()
                StatisticConsumer.producer_end_number += 1
                if StatisticConsumer.producer_end_number >= self.producer_number:
                    self.lock.release()
                    break
                self.lock.release()
                self.task_queue.put(None)
            else:
                alpha = task.alpha
                beta = task.beta
                # If alpha and bata pair is not contained in the dictionary
                if (alpha, beta) not in self.bag:
                    self.bag[(alpha, beta)] = queue.PriorityQueue()
                # Put the results into the priority queue
                sta_cal_single_image_by_pixel(task.clear_arr, task.result_arr, task.trans_arr,
                                              self.bag[(alpha, beta)])
                # time.sleep(0.0001)  # Sleep for 1 millisecond
        print('Statistic Consumer finish')


class PixelResult(object):
    def __init__(self, transmission, mse):
        self.transmission = transmission
        self.mse = mse

    def __lt__(self, other):
        return self.transmission < other.transmission


class ImageTask:
    def __init__(self, clear_image_arr, result_image_arr, transmission_arr, alpha, beta):
        self.clear_arr = clear_image_arr
        self.result_arr = result_image_arr
        self.trans_arr = transmission_arr
        self.alpha = alpha
        self.beta = beta


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
def sta_image_input(clear_dir, transmission_dir, result_dir):
    # First read the file names from clear directory
    clear_file_list = os.listdir(clear_dir)
    transmission_file_list = os.listdir(transmission_dir)
    result_file_list = os.listdir(result_dir)
    # Construct a dictionary for clear images
    for clear_image_name in clear_file_list:
        file_path = os.path.join(clear_dir, clear_image_name)
        clear_index = clear_image_name[0:constant. STATS_CLEAR_INDEX_BIT]
        constant.STATS_CLEAR_DICTIONARY[clear_index] = file_path

    # Construct a dictionary for transmission_images
    for transmission_image_name in transmission_file_list:
        file_path = os.path.join(transmission_dir, transmission_image_name)
        transmission_index = transmission_image_name[0:constant. STATS_TRANS_INDEX_BIT]
        constant.STATS_TRANSMISSION_DICTIONARY[transmission_index] = file_path

    for result_image_name in result_file_list:
        RESULT_IMAGE_QUEUE.put(result_image_name)
    RESULT_IMAGE_QUEUE.put(None)


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
    for i in range(constant. STATS_CHANNEL_NUM):
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
            q.put(PixelResult(transmission[h][w], sta_cal_mse_pixel(clear[h, w, :], result[h, w, :])))


def sta_read_image(clear_image_dir, result_image_dir, transmission_map_dir):
    # read clear, haze, transmission image into memory
    # Assert H and W are the same.
    # Return matrices for three images.
    clear_image = Image.open(clear_image_dir)
    clear = np.array(clear_image) / 255
    result_image = Image.open(result_image_dir)
    result = np.array(result_image) / 255
    # the Transmission map might save as .npy file
    if transmission_map_dir.endswith(".npy"):
        transmission = np.load(transmission_map_dir)
    else:
        transmission_image = Image.open(transmission_map_dir)
        transmission = np.array(transmission_image)
    return clear, result, transmission


def sta_group_count_average(low, group_length, result_list, final_result_map):
    up = low + group_length
    total_mse = 0.0
    t = result_list[low].transmission
    while low < up:
        single_pixel_result = result_list[low]
        total_mse += single_pixel_result.mse
        low += 1
    final_result_map[t] = total_mse/group_length


def sta_create_visual_result(result_map, pool, double_map):
    task_list = []
    # result_map is a dictionary used to save result for different alpha and beta
    for key in result_map.keys():
        single_key_list = result_map[key]   # elements in single_key_list shares same alpha and beta
        double_map[key] = {}    # {key:{}, {}}, initialization
        result_len = len(single_key_list)
        '''
               How are lists implemented?
           Python�s lists are really variable-length arrays, not Lisp-style linked lists. The implementation uses a contiguous array of references to other objects, and keeps a pointer to this array and the array�s length in a list head structure.
           This makes indexing a list a[i] an operation whose cost is independent of the size of the list or the value of the index.
           When items are appended or inserted, the array of references is resized. Some cleverness is applied to improve the performance of appending items repeatedly; when the array must be grown, some extra space is allocated so the next few times don�t require an actual resize.
        '''
        group_length = math.floor(result_len / constant. STATS_GROUP_NUM)
        for i in range(constant. STATS_GROUP_NUM):
            low = group_length * i
            lst_vars = [low, group_length, single_key_list, double_map[key]]
            func_var = [(lst_vars, None)]
            task_list.append(threadpool.makeRequests(sta_group_count_average, func_var))
        for requests in task_list:
            [pool.putRequest(req) for req in requests]
        pool.wait()


def sta_single_queue_2_list(pq, dump_list):
    while not pq.empty():
        pixel_result = pq.get()
        dump_list.append(pixel_result)
        del pixel_result    # gc
    del pq


def sta_queue_2_list(result_map, serialization_map,  pool):
    task_list = []
    for key in result_map.keys():
        serialization_map[key] = []
        lst_vars = [result_map[key], serialization_map[key]]
        func_var = [(lst_vars, None)]
        task_list.append(threadpool.makeRequests(sta_single_queue_2_list, func_var))
    for requests in task_list:
        [pool.putRequest(req) for req in requests]
    pool.wait()


def main():
    # Group order: 0, 1, 2 ... GROUP_NUM-1
    cpu_number = multiprocessing.cpu_count()
    serialization_bag = {}
    # Use the data from calculation or serialization file to create the statistical result
    # Create a thread pool, # of thread = GROUP_NUM * 2.
    pool = threadpool.ThreadPool(constant. STATS_GROUP_NUM * 4)
    temp_result_bag = {}
    if constant. STATS_START_CALCULATION:
        # call dehazenet_input to read the images directory.
        sta_image_input(constant. STATS_CLEAR_DIR, constant. STATS_TRANSMISSION_DIR, constant. STATS_RESULT_DIR)
        task_queue = queue.Queue()
        thread_list = []
        #  Start doing statistic calculation
        for producer_id in range(int(cpu_number)):
            statistic_producer = StatisticProducer(task_queue, RESULT_IMAGE_QUEUE)
            statistic_producer.start()
            thread_list.append(statistic_producer)

        time.sleep(0.0001)
        consumer_static_lock = threading.Lock()
        for consumer_id in range(cpu_number):
            statistic_consumer = StatisticConsumer(task_queue, cpu_number, consumer_static_lock, temp_result_bag)
            statistic_consumer.start()
            thread_list.append(statistic_consumer)

        for thread in thread_list:
            thread.join()
        del task_queue
        print('Step 1 : Producer-Consumer model calculation finish, Start doing statistical calculation.')

        # For each of the items in TEMP_RESULT_BAG, put items in PriorityQueue to a list for serialization.
        sta_queue_2_list(temp_result_bag, serialization_bag, pool)
        del temp_result_bag
        print("Step 2 : Finish copying queue to the list.")

        # Serialization the priority queue to SERIALIZATION_FILE_NAME
        if constant. STATS_NEED_SERIALIZATION:
            serialization_file_list = []
            for key in serialization_bag.keys():
                serialization_file_name = './alpha_' + str(key[0]) + '_beta_' + str(key[1]) + '.pkl'
                serialization_file_list.append(serialization_file_name)
            for filename in serialization_file_list:
                if os.path.exists(filename):
                    os.remove(filename)
                with open(filename, 'wb') as f:
                    pickle.dump(serialization_bag, f)  # Dump the queue into file
                f.close()
        print("Step 2.1 : Finish dump list to file.")

    else:
        # Load the queue from file
        print("Step 1 : Start loading dump file.")
        if not os.path.exists(constant. STATS_SERIALIZATION_FILE_NAME):
            raise RuntimeError("Serialization file does not exist!")
        else:
            with open(constant. STATS_SERIALIZATION_FILE_NAME, 'rb') as f:
                serialization_bag = pickle.load(f)  # load result map from file
            f.close()
            print("Step 2 : Finish loading the list from .pkl file.")

    final_result_double_map = {}    # key is the lowest transmission in current group, value is average mse.
    sta_create_visual_result(serialization_bag, pool, final_result_double_map)
    del serialization_bag
    print("Step 3 : Finish calculating visual result.")

    for key in final_result_double_map.keys():  # key is (alpha, beta)
        single_result_map = final_result_double_map[key]
        result_queue = queue.PriorityQueue()
        for k in single_result_map.keys():
            single_column = PixelResult(k, single_result_map[k])
            result_queue.put(single_column)
        print('alpha: ' + str(key[0]) + ' ;beta: ' + str(key[1]))
        while not result_queue.empty():
            result_column = result_queue.get()
            print(str(result_column.transmission) + ": " + str(result_column.mse))


if __name__ == '__main__':
    main()
