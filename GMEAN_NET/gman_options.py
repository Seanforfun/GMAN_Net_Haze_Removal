#  ====================================================
#   Filename: gman_options.py
#   Function: This file is used to do several options using transmission.
#  ====================================================
import numpy as np
import queue
import os
import multiprocessing
import threading
from PIL import Image as Image
import gman_optimize as do
import gman_transmission as dt
import WRLock
import time
import matplotlib.pyplot as plt
import shutil
from enum import Enum
from abc import ABCMeta,abstractmethod
from DarkChannelPrior import dehazenet_darkchannel as dd
from ColorAttenuationPriorDehazing import runDehazing
import gman_constant as constant


START_CONDITION = threading.Condition()
RESULT_QUEUE = queue.Queue()


class Options(Enum):
    GET_CLOSE_ZERO_TRANSMISSION_STATISTICS = 0
    GET_HISTOGRAM_WITH_CLOSE_RGB = 1
    GET_TRANSMISSION_HISTOGRAM = 2
    DEHAZE_WITH_TRANSMISSION_MAP = 3
    GET_PIXEL_NUMBER_CLOSE_FOR_LOW_TRANSMISSION = 4
    GET_ESTIMATE_ALPHA = 5


class OptionsMap(Enum):
    TRANSMISSION_MAP = 0
    DARK_CHANNEL_MAP = 1
    ATTENUATION_DEPTH_MAP = 2


class OptionsHighestIntensity(Enum):
    HIGHEST_INTENSITY = 0
    SMALLEST_DIFFERENCE_BETWEEN_CHANNELS = 1
    HIGHEST_CHANNEL_VALUE = 2


class OptionsInOutDoor(Enum):
    INDOOR = 0
    OUTDOOR = 1


# TODO Modify options here
OPTION = Options.GET_ESTIMATE_ALPHA

# TODO When Options is GET_ESTIMATE_ALPHA, Select a map used to calculate the alpha
MAP_OPTION = OptionsMap.ATTENUATION_DEPTH_MAP

# TODO Option of selecting points from 0.1 % transmission map
HIGHEST_INTENSITY_OPTION = OptionsHighestIntensity.HIGHEST_CHANNEL_VALUE

# TODO Option of indoor or outdoor
IN_OUT_DOOR = OptionsInOutDoor.OUTDOOR


class OptionFactory:
    @staticmethod
    def get_option_instance(option):
        if option == Options.GET_CLOSE_ZERO_TRANSMISSION_STATISTICS:
            return OptionDoCountCloseZero()
        elif option == Options.GET_HISTOGRAM_WITH_CLOSE_RGB:
            return OptionGetThreeChannelValueClose()
        elif option == Options.GET_TRANSMISSION_HISTOGRAM:
            return OptionGetTransmissionHistogram()
        elif option == Options.DEHAZE_WITH_TRANSMISSION_MAP:
            return OptionDehazeUsingTransmissionMap()
        elif option == Options.GET_PIXEL_NUMBER_CLOSE_FOR_LOW_TRANSMISSION:
            return OptionCheckDistancesWithSmallTransmission()
        elif option == Options.GET_ESTIMATE_ALPHA:
            return OptionGetEstimateAlpha()
        else:
            raise NotImplementedError("Method is not implemented!")

    @staticmethod
    def get_matplotlib_instance(option):
        if option == Options.GET_CLOSE_ZERO_TRANSMISSION_STATISTICS:
            return OptionDoCountCloseZero()
        elif option == Options.GET_HISTOGRAM_WITH_CLOSE_RGB:
            return OptionGetThreeChannelValueClose()
        elif option == Options.GET_TRANSMISSION_HISTOGRAM:
            return OptionGetTransmissionHistogram()
        elif option == Options.DEHAZE_WITH_TRANSMISSION_MAP:
            return OptionDehazeUsingTransmissionMap()
        elif option == Options.GET_PIXEL_NUMBER_CLOSE_FOR_LOW_TRANSMISSION:
            return OptionCheckDistancesWithSmallTransmission()
        elif option == Options.GET_ESTIMATE_ALPHA:
            return OptionGetEstimateAlpha()
        else:
            raise NotImplementedError("Method is not implemented!")


class IOption(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def do_option(self, transmission_array, tranmission_name, result_queue):
        pass


class IOptionPlot(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def print_visual_result(self):
        pass


class OptionDoCountCloseZero(IOption, IOptionPlot):
    # Get the number of transmissions under transmission threshold
    def do_option(self, transmission_array, transmission_array_name, result_queue):
        if OPTION != Options.GET_CLOSE_ZERO_TRANSMISSION_STATISTICS:
            return
        haze_arr = option_get_haze_array_with_transmission_name(transmission_array_name)
        pq = queue.Queue()
        shape = np.shape(transmission_array)
        H = shape[0]
        W = shape[1]
        size = H * W
        count = 0
        for h in range(H):
            for w in range(W):
                if transmission_array[h][w] < constant.OPTIONS_TRANSMISSION_THRESHOLD:
                    # single_result = (haze_arr[h][w][0] + haze_arr[h][w][1] + haze_arr[h][w][2]) / 3
                    print("(" + str(round(haze_arr[h][w][0], 3)) + "  " + str(round(haze_arr[h][w][1], 3)) + "  " + str(
                        round(haze_arr[h][w][2], 3)) + "), t: " + str(transmission_array[h][w]))
                    count += 1
        print("Total size: " + str(size) + " Close Zero: " + str(count))

    def print_visual_result(self):
        pass


# Calculate three channels value difference and get statistical calculation for all channels
# Put result into a queue and use matplotlib to generate the historgram
class OptionGetThreeChannelValueClose(IOption, IOptionPlot):
    def do_option(self, transmission_array, transmission_array_name, result_queue):
        if OPTION != Options.GET_HISTOGRAM_WITH_CLOSE_RGB:
            return
        haze_arr = option_get_haze_array_with_transmission_name(transmission_array_name)
        _, alpha, _ = dt.trans_get_alpha_beta(transmission_array_name)
        pq = queue.PriorityQueue()
        shape = np.shape(haze_arr)
        H = shape[0]
        W = shape[1]
        expected_number = 0
        for h in range(H):
            for w in range(W):
                if abs(haze_arr[h][w][0] - haze_arr[h][w][1]) <= constant.OPTIONS_THRESHOLD and abs(
                        haze_arr[h][w][1] - haze_arr[h][w][2]) <= constant.OPTIONS_THRESHOLD and abs(
                    haze_arr[h][w][0] - haze_arr[h][w][2]) \
                        <= constant.OPTIONS_THRESHOLD:
                    if haze_arr[h][w][0] >= constant.OPTIONS_LOWER_BOUNDARY:
                        # pq.put((haze_arr[h][w][0] + haze_arr[h][w][1] + haze_arr[h][w][2]) / 3)
                        pq.put(haze_arr[h][w][0])
                        if transmission_array[h][w] < constant.OPTIONS_TRANSMISSION_THRESHOLD:
                            expected_number += 1
        result_queue.put((alpha, pq, expected_number, transmission_array_name))

    def print_visual_result(self):
        if OPTION != Options.GET_HISTOGRAM_WITH_CLOSE_RGB:
            return
        while not RESULT_QUEUE.empty():
            s_queue = RESULT_QUEUE.get()
            alpha_gt = s_queue[0]
            plt.xlim(constant.OPTIONS_LOWER_BOUNDARY, 1)
            plt.xlabel('Hazy pixel value')
            my_x_ticks = np.arange(constant.OPTIONS_LOWER_BOUNDARY, 1, constant.OPTIONS_STEP_SIZE)
            plt.xticks(my_x_ticks)
            plt.ylabel('Number of points in this region')
            bar_list = []
            while not s_queue[1].empty():
                bar_list.append(round(s_queue[1].get(), 3))
            result_array = np.asarray(bar_list)
            size = np.size(result_array)
            plt.title("gt: " + str(alpha_gt) + "|threshold: " + str(constant.OPTIONS_THRESHOLD) + "|TransmissionThreshold: " +
                      str(constant.OPTIONS_TRANSMISSION_THRESHOLD) + "|fraction: " + str(s_queue[2]) + "/" + str(size))
            plt.hist(result_array, bins=30, width=0.006, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
            _, filename = os.path.split(s_queue[3])
            fname, _ = os.path.splitext(filename)
            fname_with_ext = fname + constant.OPTIONS_IMAGE_SUFFIX
            plt.savefig(os.path.join("./StatisticalFigure", fname_with_ext))
            plt.close()


class OptionGetTransmissionHistogram(IOption, IOptionPlot):
    # Get histogram of transmission map
    def do_option(self, transmission_array, transmission_array_name, result_queue):
        if OPTION != Options.GET_TRANSMISSION_HISTOGRAM:
            return
        _, alpha, _ = dt.trans_get_alpha_beta(transmission_array_name)
        shape = np.shape(transmission_array)
        H = shape[0]
        W = shape[1]
        expected_number = 0
        pq = queue.PriorityQueue()
        for h in range(H):
            for w in range(W):
                pq.put(transmission_array[h][w])
                if transmission_array[h][w] < constant.OPTIONS_TRANSMISSION_THRESHOLD:
                    expected_number += 1
        result_queue.put((alpha, pq, expected_number, transmission_array_name))

    def print_visual_result(self):
        if OPTION != Options.GET_TRANSMISSION_HISTOGRAM:
            return
        while not RESULT_QUEUE.empty():
            s_queue = RESULT_QUEUE.get()
            alpha_gt = s_queue[0]
            plt.xlim(0, 1)
            plt.xlabel('Hazy pixel value')
            my_x_ticks = np.arange(0, 1, constant.OPTIONS_STEP_SIZE)
            plt.xticks(my_x_ticks)
            plt.ylabel('Number of points in this region')
            bar_list = []
            while not s_queue[1].empty():
                bar_list.append(round(s_queue[1].get(), 3))
            result_array = np.asarray(bar_list)
            size = np.size(result_array)
            plt.title("gt: " + str(alpha_gt) + "|threshold: " + str(constant.OPTIONS_THRESHOLD) + "|TransmissionThreshold: " +
                      str(constant.OPTIONS_TRANSMISSION_THRESHOLD) + "|fraction: " + str(s_queue[2]) + "/" + str(size))
            plt.hist(result_array, bins=30, width=0.005, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
            _, filename = os.path.split(s_queue[3])
            fname, _ = os.path.splitext(filename)
            fname_with_ext = fname + ".png"
            plt.savefig(os.path.join("./StatisticalFigure", fname_with_ext))
            plt.close()


class OptionDehazeUsingTransmissionMap(IOption, IOptionPlot):
    # Dehaze hazy images using transmission map
    def do_option(self, transmission_array, transmission_name, result_queue):
        if OPTION != Options.DEHAZE_WITH_TRANSMISSION_MAP:
            return
        haze_arr = option_get_haze_array_with_transmission_name(transmission_name)
        # Get alpha from name
        _, alpha, _ = dt.trans_get_alpha_beta(transmission_name)
        do.opt_write_result_to_file(do.opt_dehaze_with_alpha_transmission(alpha, transmission_array, haze_arr))

    def print_visual_result(self):
        pass


class OptionCheckDistancesWithSmallTransmission(IOption, IOptionPlot):
    # Check if the t restriction is satisfied, and check how many pixels match the three channel values are close enough
    def do_option(self, transmission_array, transmission_name, result_queue):
        if OPTION != Options.GET_PIXEL_NUMBER_CLOSE_FOR_LOW_TRANSMISSION:
            return
        haze_arr = option_get_haze_array_with_transmission_name(transmission_name)
        shape = np.shape(transmission_array)
        H = shape[0]
        W = shape[1]
        number_counter = 0
        for h in range(H):
            for w in range(W):
                if transmission_array[h][w] < constant.OPTIONS_TRANSMISSION_THRESHOLD:
                    if abs(haze_arr[h][w][0] - haze_arr[h][w][1]) < constant.OPTIONS_THRESHOLD and \
                            abs(haze_arr[h][w][1] - haze_arr[h][w][2]) < constant.OPTIONS_THRESHOLD and \
                            abs(haze_arr[h][w][0] - haze_arr[h][w][2]) < constant.OPTIONS_THRESHOLD:
                        number_counter += 1
        print(number_counter)

    def print_visual_result(self):
        pass


class OptionMapFactory():

    def __init__(self, option_map):
        self.option_map = option_map

    def get_map(self, haze_arr):
        if self.option_map == OptionsMap.TRANSMISSION_MAP:
            return OptionMapFactory.__get_transmission_map(haze_arr)
        elif self.option_map == OptionsMap.DARK_CHANNEL_MAP:
            return OptionMapFactory.__get_dark_channel_transmission_map(haze_arr)
        elif self.option_map == OptionsMap.ATTENUATION_DEPTH_MAP:
            return OptionMapFactory.__get_attenuation_depth_map(haze_arr)
        else:
            raise ValueError("Not Implemented map type!")

    @staticmethod
    def __get_transmission_map(haze_arr):
        pass

    @staticmethod
    def __get_dark_channel_transmission_map(haze_arr):
        u8_haze_arr = haze_arr.astype("uint8")
        u64_haze_arr = haze_arr.astype("float64")
        dark = dd.DarkChannel(u64_haze_arr, 15)
        alpha = dd.AtmLight(u64_haze_arr, dark)
        te = dd.TransmissionEstimate(u64_haze_arr, alpha, 15)
        return dd.TransmissionRefine(u8_haze_arr, te)

    @staticmethod
    def __get_attenuation_depth_map(haze_arr):
        dR, _ = runDehazing.calDepthMap((haze_arr * 255).astype('uint8'), 15)
        guided_filter = runDehazing.GuidedFilter(haze_arr, 60, 10 ** -3)
        refineDR = guided_filter.filter(dR)
        tR = np.exp(-1.0 * refineDR)
        return tR


class OptionGetEstimateAlpha(IOption, IOptionPlot):

    class Pixel(object):
        def __init__(self, transmission, h, w):
            self.transmission = transmission
            self.h = h
            self.w = w

        def __lt__(self, other):
            return self.transmission < other.transmission

    class ChannelDistance(object):
        def __init__(self, distance, h, w):
            self.distance = distance
            self.h = h
            self.w = w

        def __lt__(self, other):
            return self.distance < self.distance

    def do_option(self, transmission_array, transmission_name, result_queue):
        if OPTION != Options.GET_ESTIMATE_ALPHA:
            return
        _, alpha, _ = dt.trans_get_alpha_beta(transmission_name)
        haze_arr = option_get_haze_array_with_transmission_name(transmission_name)
        intermediate_map = OptionMapFactory(MAP_OPTION).get_map(haze_arr)
        # attenuation_depth_map = OptionGetEstimateAlpha.__option_get_depth_color_attenuation(haze_arr)
        OptionGetEstimateAlpha.__estimate_alpha_with_map(intermediate_map, haze_arr, alpha, result_queue)

    def print_visual_result(self):
        if OPTION != Options.GET_ESTIMATE_ALPHA:
            return
        sum_error_rate = 0
        size = RESULT_QUEUE.qsize()
        while not RESULT_QUEUE.empty():
            sum_error_rate += RESULT_QUEUE.get()
        print("Error rate: " + '{:.3%}'.format(sum_error_rate / size))

    @staticmethod
    def __estimate_alpha_with_map(transmission, haze, alpha, result_queue):
        pq = queue.PriorityQueue()
        shape = np.shape(transmission)
        H = shape[0]
        W = shape[1]
        point_one_number = int(np.size(transmission) * 0.001)
        for h in range(H):
            for w in range(W):
                pq.put(OptionGetEstimateAlpha.Pixel(transmission[h][w], h, w))
        estimate_alpha = OptionGetEstimateAlpha.__get_estimate_alpha_by_option(HIGHEST_INTENSITY_OPTION,
                                                                               pq, point_one_number, haze)
        printed_estimate_alpha = abs(estimate_alpha - float(alpha)) / float(alpha)
        print("GT: %.3f Estimate Alpha: %.5f Error rate: " % (float(alpha), estimate_alpha) + '{:.3%}'.format(
            printed_estimate_alpha))
        result_queue.put(printed_estimate_alpha)

    @staticmethod
    def __get_estimate_alpha_by_option(option, transmission_queue, point_number, haze_array):
        '''
        :param option:  Decide using what kind of method to estimate alpha.
        :param transmission_queue: In order to get the 0.1% smallest medium map value and get estimate alpha
        :param point_number: Number of points for 0.1% of medium map
        :param haze_array: RGB array of hazed images.
        :return: Estimate alpha
        '''
        if option == OptionsHighestIntensity.SMALLEST_DIFFERENCE_BETWEEN_CHANNELS:
            return OptionGetEstimateAlpha.__get_estimate_alpha_by_smallest_distance(transmission_queue, point_number, haze_array)
        elif option == OptionsHighestIntensity.HIGHEST_INTENSITY:
            return OptionGetEstimateAlpha.__get_estimate_alpha_by_highest_average(transmission_queue, point_number, haze_array)
        elif option == OptionsHighestIntensity.HIGHEST_CHANNEL_VALUE:
            return OptionGetEstimateAlpha.__get_estimate_alpha_by_highest_channel_value(transmission_queue, point_number, haze_array)
        else:
            raise ValueError("Current option is not implemented!")

    @staticmethod
    def __get_estimate_alpha_by_smallest_distance(transmission_queue, point_number, haze):
        pq_for_minimum_distance = queue.PriorityQueue()
        while point_number > 0:
            point_number -= 1
            pixel = transmission_queue.get()
            distance = (haze[pixel.h][pixel.w][0] - haze[pixel.h][pixel.w][1]) ** 2 + \
                       (haze[pixel.h][pixel.w][1] - haze[pixel.h][pixel.w][2]) ** 2 + \
                       (haze[pixel.h][pixel.w][0] - haze[pixel.h][pixel.w][2]) ** 2
            pq_for_minimum_distance.put(OptionGetEstimateAlpha.ChannelDistance(distance, pixel.h, pixel.w))
        solution_pixel = pq_for_minimum_distance.get()
        return max(haze[solution_pixel.h][solution_pixel.w][0],  haze[solution_pixel.h][solution_pixel.w][1]
                          , haze[solution_pixel.h][solution_pixel.w][2])

    @staticmethod
    def __get_estimate_alpha_by_highest_average(transmission_queue, point_number, haze):
        maximum_intensity = 0
        while point_number > 0:
            point_number -= 1
            pixel = transmission_queue.get()
            maximum_intensity = max(
                ((haze[pixel.h][pixel.w][0] + haze[pixel.h][pixel.w][1] + haze[pixel.h][pixel.w][2]) / 3),
                maximum_intensity)
        return maximum_intensity

    @staticmethod
    def __get_estimate_alpha_by_highest_channel_value(transmission_queue, point_number, haze):
        estiamte_alpha = 0
        while point_number > 0:
            point_number -= 1
            pixel = transmission_queue.get()
            estiamte_alpha = max(np.max(haze[pixel.h, pixel.w, :]), estiamte_alpha)
        return estiamte_alpha


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
            if IN_OUT_DOOR == OptionsInOutDoor.OUTDOOR:
                arr = np.load(t)
            elif IN_OUT_DOOR == OptionsInOutDoor.INDOOR:
                # open transmission from knight file
                arr = np.array(Image.open(t)) / 255
            else:
                raise RuntimeError("Indoor or outdoor option is not implemented!")
            self.task_queue.put((arr, t))
            # if START_CONDITION.acquire():
            #     START_CONDITION.notify_all()
            # START_CONDITION.release()
        print('Producer finish')

    @staticmethod
    def __indoor_get_transmission_by_name(self, name):
        return np.array(Image.open(name)) / 255


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
                option_operator = OptionFactory.get_option_instance(OPTION)
                option_operator.do_option(task[0], task[1], RESULT_QUEUE)
        print('Consumer finish')


# Get normalize haze array
def option_get_haze_array_with_transmission_name(name):
    _, filename = os.path.split(name)
    fname, _ = os.path.splitext(filename)
    fname_with_ext = fname + constant.OPTIONS_IMAGE_SUFFIX
    full_name = os.path.join(constant.OPTIONS_HAZY_DIR, fname_with_ext)
    return np.array(Image.open(full_name)) / 255


def option_input(t_dir):
    t_file_list = os.listdir(t_dir)
    q = queue.Queue()
    for filename in t_file_list:
        q.put(os.path.join(t_dir, filename))
    q.put(None)
    return q


def main():
    if os.path.exists(constant.OPTIONS_STATISTICAL_DIR):
        shutil.rmtree(constant.OPTIONS_STATISTICAL_DIR)
        os.mkdir(constant.OPTIONS_STATISTICAL_DIR)
    q = option_input(constant.OPTIONS_TRANS_DIR)
    cpu_num = multiprocessing.cpu_count() * 2
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

    plot_instance = OptionFactory.get_matplotlib_instance(OPTION)
    if plot_instance is not None:
        plot_instance.print_visual_result()


if __name__ == '__main__':
    main()
