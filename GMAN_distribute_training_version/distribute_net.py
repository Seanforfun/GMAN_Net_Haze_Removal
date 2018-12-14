#  ====================================================
#   Filename: distribute_net.py
#   Function: This file is used to define the CNN network structure
#   actions, users can override pre_process and post_process to
#   add aspects for the forward processing.
#  ====================================================


def current_model(**kwds):
    def decorate(f):
        for k in kwds:
            if k == 'net':
                setattr(f, k, kwds[k])
        return f
    return decorate


class Net(object):
    def __init__(self, model=None):
        if model is not None:
            self.model = model

    def inference(self, pre_proccessed_data):
        assert self.model is not None, "Please either create a model or use annotation @current_model"
        return self.model.inference(pre_proccessed_data)

    @staticmethod
    def pre_process(input_data, *args, **kwargs):
        return input_data

    @staticmethod
    def post_process(result, *args, **kwargs):
        return result

    def process(self, input_data, *args, **kwargs):
        pre_processed_data = Net.pre_process(input_data, args, kwargs)
        result = self.inference(pre_processed_data)
        return Net.post_process(result, args, kwargs)


if __name__ == '__main__':
    pass
