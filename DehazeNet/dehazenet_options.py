import numpy as np
import queue
import os
import multiprocessing
import threading
from PIL import Image as Image
import dehazenet_optimize as do
import dehazenet_transmission as dt
import WRLock
import time
import matplotlib.pyplot as plt
import dehazenet_transmission as dtrans

TRANS_DIR = "./ClearImages/TransImages"
HAZY_DIR = "./HazeImages/TestImages"
START_CONDITION = threading.Condition()
RESULT_QUEUE = queue.Queue()
THRESHOLD = 0.01
LOWER_BOUNDARY = 0.7
STEP_SIZE = 0.01
TRANSMISSION_THRESHOLD = 0.01


# task[0]: Transmission array
# task[1]: Name of the transmission map
class OptionsProducer(threading.Thread):
    def __init__(self, input_queue, task_queue):
        threading.Thread.__init__(self)
        self.queue = input_queue
        self.task_queue = task_queue

    def run(self):
        while True:
            t = self.queue.get()
            if t is None:
                self.queue.put(None)
                self.task_queue.put(None)
                break
            arr = np.load(t)
            self.task_queue.put((arr, t))
            # if START_CONDITION.acquire():
            #     START_CONDITION.notify_all()
            # START_CONDITION.release()
        print('Producer finish')


class OptionsConsumer(threading.Thread):
    producer_end_number = 0

    def __init__(self, task_queue, lock, producer_num):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.lock = lock
        self.producer_num = producer_num

    def run(self):
        # if START_CONDITION.acquire():
        #     START_CONDITION.wait()
        # START_CONDITION.release()
        while True:
            task = self.task_queue.get()
            if task is None:
                self.lock.acquire_write()
                OptionsConsumer.producer_end_number += 1
                self.lock.demote()
                if OptionsConsumer.producer_end_number > self.producer_num:
                    self.lock.release()
                    break
                self.lock.release()
                self.task_queue.put(None)
            else:
                option_do_count_close_zero(task[0], task[1], RESULT_QUEUE)

                # TODO: Calculate three channels value difference and get statistical calculation for all channels
                # option_get_three_channel_value_close(task[0], task[1], RESULT_QUEUE)

                # TODO: (Optional) Dehaze hazy images using transmission map
                # result = option_dehaze_using_transmission_map(task[0], task[1])
                # do.opt_write_result_to_file(result)

                # TODO: Check if the t restriction is satisfied, if the three channel values are close enough
                # option_check_distances_with_small_transmission(task[0], task[1])
        print('Consumer finish')


def option_check_distances_with_small_transmission(transmission, transmission_name):
    haze_arr = option_get_haze_array_with_transmission_name(transmission_name)
    shape = np.shape(transmission)
    H = shape[0]
    W = shape[1]
    number_counter = 0
    for h in range(H):
        for w in range(W):
            if transmission[h][w] < TRANSMISSION_THRESHOLD:
                if abs(haze_arr[h][w][0] - haze_arr[h][w][1]) < THRESHOLD and \
                        abs(haze_arr[h][w][1] - haze_arr[h][w][2]) < THRESHOLD and\
                        abs(haze_arr[h][w][0] - haze_arr[h][w][2]) < THRESHOLD:
                    number_counter += 1
    print(number_counter)


def option_get_three_channel_value_close(transmission_array, transmission_array_name, result_queue):
    haze_arr = option_get_haze_array_with_transmission_name(transmission_array_name)
    _, alpha, _ = dt.trans_get_alpha_beta(transmission_array_name)
    pq = queue.PriorityQueue()
    shape = np.shape(haze_arr)
    H = shape[0]
    W = shape[1]
    expected_number = 0
    for h in range(H):
        for w in range(W):
            if abs(haze_arr[h][w][0] - haze_arr[h][w][1]) < THRESHOLD and \
                    abs(haze_arr[h][w][1] - haze_arr[h][w][2]) < THRESHOLD and abs(haze_arr[h][w][0] - haze_arr[h][w][2]) < THRESHOLD\
                    and haze_arr[h][w][1] >= 0 and haze_arr[h][w][0] > LOWER_BOUNDARY:
                pq.put((haze_arr[h][w][0] + haze_arr[h][w][1] + haze_arr[h][w][2]) / 3)
                if transmission_array[h][w] < TRANSMISSION_THRESHOLD:
                    expected_number += 1
    result_queue.put((alpha, pq, expected_number, transmission_array_name))


# Get normalize haze array
def option_get_haze_array_with_transmission_name(name):
    _, filename = os.path.split(name)
    fname, _ = os.path.splitext(filename)
    fname_with_ext = fname + ".jpg"
    full_name = os.path.join(HAZY_DIR, fname_with_ext)
    return np.array(Image.open(full_name)) / 255


def option_dehaze_using_transmission_map(transmission, transmission_name):
    haze_arr = option_get_haze_array_with_transmission_name(transmission_name)
    # Get alpha from name
    _, alpha, _ = dt.trans_get_alpha_beta(transmission_name)
    return do.opt_dehaze_with_alpha_transmission(alpha, transmission, haze_arr)


def option_do_count_close_zero(transmission_array, transmission_array_name, result_queue):
    haze_arr = option_get_haze_array_with_transmission_name(transmission_array_name)
    pq = queue.Queue()
    shape = np.shape(transmission_array)
    H = shape[0]
    W = shape[1]
    size = H * W
    count = 0
    for h in range(H):
        for w in range(W):
            if transmission_array[h][w] < TRANSMISSION_THRESHOLD:
                single_result = (haze_arr[h][w][0] + haze_arr[h][w][1] + haze_arr[h][w][2]) / 3
                print("(" + str(round(haze_arr[h][w][0], 3)) + "  " + str(round(haze_arr[h][w][1], 3)) + "  " + str(round(haze_arr[h][w][2], 3)) + ")")
                count += 1
    print("Total size: " + str(size) + " Close Zero: " + str(count))


def option_input(t_dir):
    t_file_list = os.listdir(t_dir)
    q = queue.Queue()
    for filename in t_file_list:
        q.put(os.path.join(t_dir, filename))
    q.put(None)
    return q


def main():
    q = option_input(TRANS_DIR)
    cpu_num = multiprocessing.cpu_count()
    task_queue = queue.Queue()
    thread_list = []
    flag_lock = WRLock.RWLock()
    for producer_id in range(cpu_num):
        producer = OptionsProducer(q, task_queue)
        producer.start()
        thread_list.append(producer)

    time.sleep(0.0001)
    for consumer_id in range(cpu_num):
        consumer = OptionsConsumer(task_queue, flag_lock, cpu_num)
        consumer.start()
        thread_list.append(consumer)
    for t in thread_list:
        t.join()

    while not RESULT_QUEUE.empty():
        s_queue = RESULT_QUEUE.get()
        alpha_gt = s_queue[0]
        plt.xlabel('Hazy pixel value')
        # plt.xlim((0.7, 1))
        my_x_ticks = np.arange(LOWER_BOUNDARY, 1, STEP_SIZE)
        plt.xticks(my_x_ticks)
        plt.ylabel('Number of points in this region')
        # single_result = "gt: " + str(alpha_gt) + "| values:"
        # bar list saves all point numbers in all regions
        bar_list = []
        while not s_queue[1].empty():
            # single_result += ' ' + str(round(s_queue[1].get(), 3))
            bar_list.append(s_queue[1].get())
        result_array = np.asarray(bar_list)
        size = np.size(result_array)
        plt.title("gt: " + str(alpha_gt) + "|threshold: " + str(THRESHOLD) + "|TransmissionThreshold: " +
                  str(TRANSMISSION_THRESHOLD) + "|fraction: " + str(s_queue[2]) + "/" + str(size))
        plt.hist(result_array, bins=30, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        _, filename = os.path.split(s_queue[3])
        fname, _ = os.path.splitext(filename)
        fname_with_ext = fname + ".png"
        plt.savefig(os.path.join("./StatisticalFigure", fname_with_ext))


if __name__ == '__main__':
    main()