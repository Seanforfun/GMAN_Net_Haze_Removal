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

# TODO Need to fill the directory
CLEAR_DIR = './ClearImages/TestImages'
HAZY_DIR = './HazeImages/TestImages'
TRANSMISSION_DIR = './ClearImages/TransImages'
# Dictinary for saving clear images
CLEAR_IMAGE_DICTIONARY = {}
HAZY_IMAGE_LIST = []

IMAGE_INDEX = 0
IMAGE_A = 1
IMAGE_BETA = 2

PRODUCER_FINISH = False
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
    return np.array(image)


class TransProducer(threading.Thread):
    # Add tasks into task queue.
    def __init__(self, hazy_dir, task_queue, wrlock):
        threading.Thread.__init__(self)
        self.hazy_dir = hazy_dir
        self.queue = queue
        self.task_queue = task_queue
        self.wrlock = wrlock

    def run(self):
        file_list = os.listdir(self.hazy_dir)
        for hazy_image in file_list:
            filename, file_extension = os.path.splitext(hazy_image)
            hazed_image_split = filename.split('_')
            clear_index = hazed_image_split[IMAGE_INDEX]
            image_alpha = hazed_image_split[IMAGE_A]
            image_beta = hazed_image_split[IMAGE_BETA]
            clear_image_path = CLEAR_IMAGE_DICTIONARY[clear_index]
            hazy_image_path = os.path.join(self.hazy_dir, hazy_image)
            task = Task(clear_index, float(image_alpha), float(image_beta), trans_read_image_array(clear_image_path),
                        trans_read_image_array(hazy_image_path))
            self.task_queue.put(task)
            if START_CONDITION.acquire():
                START_CONDITION.notify_all()
            START_CONDITION.release()
        # Put a exit signal into the queue to inform the producer
        self.task_queue.put(None)
        print('Producer finish')


class TransConsumer(threading.Thread):
    def __init__(self, task_queue, wrlock):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.wrlock = wrlock

    def run(self):
        if START_CONDITION.acquire():
            START_CONDITION.wait()
        START_CONDITION.release()
        while True:
            # task_queue is empty and all images are loaded(consumer end)
            task = self.task_queue.get()
            if task is None:
                self.task_queue.put(None)
                break
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
    return (hazy_array[:, :, 0] - alpha_matrix) / (clear_array[:, :, 0] - alpha_matrix)


def trans_input(clear_dir):
    clear_file_list = os.listdir(clear_dir)
    # Add clear images into dict
    for clear_image in clear_file_list:
        file_path = os.path.join(clear_dir, clear_image)
        clear_index = clear_image[0:di.IMAGE_INDEX_BIT]
        CLEAR_IMAGE_DICTIONARY[clear_index] = file_path


def main():
    # Step 1, read hazy images and clear images.
    trans_input(CLEAR_DIR)
    cpu_number = multiprocessing.cpu_count()
    # Step 2, get corresponding transmission maps and save them into directory.
    task_queue = queue.Queue()
    flag_lock = WRLock.RWLock()
    thread_list = []
    # Producer-Consumer Patterns
    #  Producer:Read images and generate tasks
    producer = TransProducer(HAZY_DIR, task_queue, flag_lock)
    producer.start()
    thread_list.append(producer)
    # Consumer: Calculate and save transmission images.
    for consumer_index in range(int(cpu_number * 2/3)):
        consumer = TransConsumer(task_queue, flag_lock)
        consumer.start()
        thread_list.append(consumer)
    for t in thread_list:
        t.join()
    print('Finish Generating transmission map!')


if __name__ == '__main__':
    main()