#  ====================================================
#   Filename: dehazenet_transmission.py
#   Function: This file is used to get the transmission map used to
#  generate hazy images from clear images.
#  ====================================================
import os
import dehazenet_input as di
import queue
import threading
from PIL import Image as Image
import numpy as np
import time
import WRLock
import multiprocessing

CLEAR_DIR = './ClearImages/TestImages'
HAZY_DIR = './HazeImages/TestImages'
TRANSMISSION_DIR = './ClearImages/TransImages'
# Dictinary for saving clear images
CLEAR_IMAGE_DICTIONARY = {}
HAZY_IMAGE_LIST = []

IMAGE_INDEX = 0
IMAGE_A = 1
IMAGE_BETA = 2

START_CONDITION = threading.Condition()


class Task:
    def __init__(self, index, a, beta, clear_image_array, hazy_image_array):
        self.index = index
        self.a = a
        self.beta = beta
        self.clear_image_array = clear_image_array
        self.hazy_image_array = hazy_image_array


def trans_read_image_array(image_path):
    image = Image.open(image_path)
    return np.array(image) / 255


def trans_get_alpha_beta(filename_with_extension):
    filename, file_extension = os.path.splitext(filename_with_extension)
    filename_split = filename.split('_')
    return filename_split[IMAGE_INDEX], filename_split[IMAGE_A], filename_split[IMAGE_BETA]


class TransProducer(threading.Thread):
    # Add tasks into task queue.
    def __init__(self, hazy_dir, task_queue, hazy_iamge_queue):
        threading.Thread.__init__(self)
        self.hazy_dir = hazy_dir
        self.queue = hazy_iamge_queue
        self.task_queue = task_queue

    def run(self):
        while True:
            hazy_dict = self.queue.get()
            if hazy_dict is None:
                self.queue.put(None)
                self.task_queue.put(None)
                break
            # Get a correct path
            clear_index, image_alpha, image_beta = trans_get_alpha_beta(os.path.basename(hazy_dict))
            clear_image_path = CLEAR_IMAGE_DICTIONARY[clear_index]
            task = Task(clear_index, float(image_alpha), float(image_beta), trans_read_image_array(clear_image_path),
                        trans_read_image_array(hazy_dict))
            self.task_queue.put(task)
            if START_CONDITION.acquire():
                START_CONDITION.notify_all()
            START_CONDITION.release()
        print('Producer finish')


class TransConsumer(threading.Thread):
    producer_end_number = 0

    def __init__(self, task_queue, lock, producer_num):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.lock = lock
        self.producer_num = producer_num

    def run(self):
        if START_CONDITION.acquire():
            START_CONDITION.wait()
        START_CONDITION.release()
        while True:
            # task_queue is empty and all images are loaded(consumer end)
            task = self.task_queue.get()
            if task is None:
                self.lock.acquire()
                TransConsumer.producer_end_number += 1
                if TransConsumer.producer_end_number >= self.producer_num:
                    self.lock.release()
                    break
                self.lock.release()
                self.task_queue.put(None)
            else:
                t = trans_get_transmission_map(task)
                np.save(os.path.join(TRANSMISSION_DIR, task.index + '_' + str(task.a) + '_' + str(task.beta) + '.npy'), t)
                time.sleep(0.001)   # Sleep for 1 millisecond
        print('Consumer finish')


def trans_get_transmission_map(task):
    clear_array = task.clear_image_array
    hazy_array = task.hazy_image_array
    shape = np.shape(clear_array)
    alpha_matrix = np.ones((shape[0], shape[1])) * task.a
    t = (hazy_array[:,:,0] - alpha_matrix) / (clear_array[:,:,0] - alpha_matrix)
    where_are_inf = np.isinf(t)
    t[where_are_inf] = 1
    where_are_nan = np.isnan(t)
    t[where_are_nan] = 0
    return t


def trans_input(clear_dir, hazy_dir):
    clear_file_list = os.listdir(clear_dir)
    hazy_file_list = os.listdir(hazy_dir)
    # Add clear images into dict
    for clear_image in clear_file_list:
        file_path = os.path.join(clear_dir, clear_image)
        clear_index = clear_image[0:di.IMAGE_INDEX_BIT]
        CLEAR_IMAGE_DICTIONARY[clear_index] = file_path

    q = queue.Queue()
    for hazy_image in hazy_file_list:
        file_path = os.path.join(hazy_dir, hazy_image)
        q.put(file_path)
    q.put(None)  # Add None as the last flag
    return q


def main():
    # Step 1, read hazy images and clear images.
    q = trans_input(CLEAR_DIR, HAZY_DIR)
    cpu_number = multiprocessing.cpu_count()
    # Step 2, get corresponding transmission maps and save them into directory.
    task_queue = queue.Queue()
    flag_lock = threading.Lock()
    thread_list = []
    # Producer-Consumer Patterns
    #  Producer:Read images and generate tasks
    for producer_id in range(cpu_number):
        producer = TransProducer(HAZY_DIR, task_queue, q)
        producer.start()
        thread_list.append(producer)
    # Consumer: Calculate and save transmission images.
    for consumer_index in range(cpu_number):
        consumer = TransConsumer(task_queue, flag_lock, cpu_number)
        consumer.start()
        thread_list.append(consumer)
    for t in thread_list:
        t.join()
    print('Finish Generating transmission map!')


if __name__ == '__main__':
    main()