#  ====================================================
#   Filename: gman_learningrate.py
#   Function: This file describes the model of learning rate, it
#   has functions of learning rate persistance(save and load).
#  ====================================================
import os
import json
import tensorflow as tf

import gman_log as logger
import gman_constant as constants
import gman_flags as flags


class LearningRate(object):
    def __init__(self, initial_learning_rate, save_path):
        self.path = save_path
        if os.path.exists(save_path):
            self.learning_rate = self.load()
        else:
            self.learning_rate = initial_learning_rate
        self.decay_factor = constants.LEARNING_RATE_DECAY_FACTOR

    def save(self, current_learning_rate):
        current_learning_rate = str(current_learning_rate)
        if not os.path.exists(self.path):
            logger.info("Create Json file for learning rate.")
        learning_rate_file = open(self.path, "w")
        try:
            learning_rate = {'learning_rate': current_learning_rate}
            json.dump(learning_rate, learning_rate_file)
        except IOError as err:
            raise RuntimeError("[Error]: Error happens when read/write " + self.path + ".")
        finally:
            learning_rate_file.close()
        return float(learning_rate["learning_rate"])

    def load(self):
            if not os.path.exists(self.path):
                return constants.INITIAL_LEARNING_RATE
            else:
                # File exist, we need to load the json object
                learning_rate_file = open(self.path, "r")
                try:
                    learning_rate_map = json.load(learning_rate_file)
                    learning_rate = learning_rate_map["learning_rate"]
                except IOError as err:
                    raise RuntimeError("[Error]: Error happens when read/write " + flags.FLAGS.train_json_path + ".")
                finally:
                    learning_rate_file.close()
                return float(learning_rate)

    def update(self, current_learning_rate):
        self.save(current_learning_rate)


if __name__ == '__main__':
     pass
