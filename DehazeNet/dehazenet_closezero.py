import numpy as np
import queue
import os
import multiprocessing
import threading

TRANS_DIR = "./ClearImages/TransImages"
START_CONDITION = threading.Condition()


class CZProducer(threading.Thread):
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
            self.task_queue.put(arr)
            if START_CONDITION.acquire():
                START_CONDITION.notify_all()
            START_CONDITION.release()
        print('Producer finish')


class CZConsumer(threading.Thread):
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
            task = self.task_queue.get()
            if task is None:
                self.lock.acquire()
                CZConsumer.producer_end_number += 1
                if CZConsumer.producer_end_number >= self.producer_num:
                    self.lock.release()
                    break
                self.lock.release()
                self.task_queue.put(None)
            else:
                # alpha = opt_find_best_alpha(task.result, task.haze, task.alpha)
                # self.result_queue.put((alpha, task.alpha))
                cz_do_count_close_zero(task)
        print('Consumer finish')


def cz_do_count_close_zero(task):
    shape = np.shape(task)
    H = shape[0]
    W = shape[1]
    size = H * W
    count = 0
    for h in range(H):
        for w in range(W):
            if task[h][w] <= 0.01:
                count += 1
    print("Total size: " + str(size) + " Close Zero: " + str(count))


def cz_input(t_dir):
    t_file_list = os.listdir(t_dir)
    q = queue.Queue()
    for filename in t_file_list:
        q.put(os.path.join(t_dir, filename))
    q.put(None)
    return q


def main():
    q = cz_input(TRANS_DIR)
    cpu_num = multiprocessing.cpu_count()
    task_queue = queue.Queue()
    thread_list = []
    flag_lock = threading.Lock()
    for producer_id in range(cpu_num):
        producer = CZProducer(q, task_queue)
        producer.start()
        thread_list.append(producer)

    for consumer_id in range(cpu_num):
        consumer = CZConsumer(task_queue, flag_lock, cpu_num)
        consumer.start()
        thread_list.append(consumer)
    for t in thread_list:
        t.join()


if __name__ == '__main__':
    main()