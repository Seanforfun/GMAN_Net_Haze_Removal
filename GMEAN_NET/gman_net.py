#  ====================================================
#   Filename: gman_net.py
#   Function: This file is used to define the CNN network structure
#   of our GMEAN net
#  ====================================================


class Net(object):
    def __init__(self, model):
        self.model = model

    def inference(self, pre_proccessed_data):
        return self.model.inference(pre_proccessed_data)

    @staticmethod
    def pre_process(input_data):
        return input_data

    @staticmethod
    def post_process(result):
        return result

    def process(self, input_data):
        pre_processed_data = Net.pre_process(input_data)
        result = self.inference(pre_processed_data)
        return Net.post_process(result)


if __name__ == '__main__':
    pass
