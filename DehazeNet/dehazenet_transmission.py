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

# TODO Need to fill the directory
CLEAR_DIR = ''
HAZY_DIR = ''
TRANSMISSION_DIR = ''
# Dictinary for saving clear images
CLEAR_IMAGE_DICTIONARY = {}
HAZY_IMAGE_LIST = []
CONSUMER_FINISH = False

IMAGE_INDEX = 0
IMAGE_A = 1
IMAGE_BETA = 2


class Task:
    def __init__(self, index, a, beta, clear_image_array, hazy_image_array):
        self.index = index
        self.a = a
        self.beta = beta
        self.clear_image_array = clear_image_array
        self.hazy_image_array = hazy_image_array


def trans_read_image_array(image_path):
    image = Image.open(image_path)
    return np.array(image)


class TransProducer(threading.Thread):
    # Add tasks into task queue.
    def __init__(self, hazy_dir, task_queue, wrlock, flag):
        threading.Thread.__init__(self)
        self.hazy_dir = hazy_dir
        self.queue = queue
        self.task_queue = task_queue
        self.flag = flag
        self.wrlock = wrlock

    def run(self):
        file_list = os.listdir(self.hazy_dir)
        for hazy_image in file_list:
            hazed_image_split = hazy_image.split('_')
            clear_index = hazed_image_split[IMAGE_INDEX]
            image_alpha = hazed_image_split[IMAGE_A]
            image_beta = hazed_image_split[IMAGE_BETA]
            clear_image_path = CLEAR_IMAGE_DICTIONARY[clear_index]
            hazy_image_path = os.path.join(self.hazy_dir, hazy_image)
            task = Task(clear_index, image_alpha, image_beta, trans_read_image_array(clear_image_path),
                        trans_read_image_array(hazy_image_path))
            self.task_queue.put(task)
        self.wrlock.write_acquire()
        self.flag = True
        self.wrlock.write_release()


class TransConsumer(threading.Thread):
    def __init__(self, task_queue, flag, wrlock):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.flag = flag
        self.wrlock = wrlock

    def run(self):
        while True:
            self.wrlock.read_acquire()
            current_flag = self.flag
            self.wrlock.read_release()
            # task_queue is empty and all images are loaded(consumer end)
            if self.task_queue.empty() and current_flag:
                return
            task = self.task_queue.get()
            t = trans_get_transmission_map(task)
            np.save(os.path.join(TRANSMISSION_DIR, task.index + '_' + task.a + '_' + task.beta + '.npy'), t)
            time.sleep(0.001)   # Sleep for 1 millisecond


def trans_get_transmission_map(task):
    clear_array = task.clear_image_array
    hazy_array = task.hazy_image_array
    shape = np.shape(clear_array)
    alpha_matrix = np.ones((shape[0], shape[1])) * task.a
    return (hazy_array[:, :, 0] - alpha_matrix) / (clear_array[:, :, 0] - alpha_matrix)


def trans_input(clear_dir):
    clear_file_list = os.listdir(clear_dir)
    # Add clear images into dict
    for clear_image in clear_file_list:
        file_path = os.path.join(clear_dir, clear_image)
        clear_index = clear_image[0:di.IMAGE_INDEX_BIT]
        CLEAR_IMAGE_DICTIONARY[clear_index] = clear_index


def main():
    # Step 1, read hazy images and clear images.
    trans_input(CLEAR_DIR)
    # Step 2, get corresponding transmission maps and save them into directory.
    task_queue = queue.Queue()
    flag_lock = WRLock.RWLock()
    # Producer-Consumer Pattern
    #  Producer:Read images and generate tasks
    producer = TransProducer(HAZY_DIR, task_queue, flag_lock, CONSUMER_FINISH)
    producer.start()
    # Consumer: Calculate and save transmission images.
    consumer = TransConsumer(task_queue, CONSUMER_FINISH, flag_lock)
    consumer.start()
    threading.current_thread().join()
    print('Finish Generating transmission map!')


if __name__ == '__main__':
    main()