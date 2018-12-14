#  ====================================================
#   Filename: distribute_experiment.py
#   Author: Botao Xiao
#   Function: This file is used to save the class of DistributeExperiment
#  ====================================================

import distribute_input as Input
import distribute_flags as flags


def current_input(**kwds):
    def decorate(f):
        for k in kwds:
            if k == 'train_input' or k == 'eval_input':
                setattr(f, k, kwds[k])
        return f
    return decorate


@current_input(train_input="", eval_input='')
class DistributeExperiment(object):
    def __init__(self, mode, train_fn=None,
                 train_dataloader=None,
                 eval_fn=None,
                 eval_dataloader=None,
                 features=None):
        self.mode = mode
        if train_fn is None and eval_fn is None:
            raise ValueError("At least provide a funtion for processing")
        if flags.FLAGS.data_load_option != 'tfrecords' and flags.FLAGS.data_load_option != 'placeholder':
            raise ValueError("Please specify a valid data load option [tfrecords, placeholder] as flag.")
        self.input_mode = Input.InputOptions.TF_RECORD if flags.FLAGS.data_load_option == 'tfrecords' \
            else Input.InputOptions.PLACEHOLDER
        if mode == "Train":
            if train_fn is None:
                raise ValueError("In Train mode, train_fn cannot be None")
            else:
                self.train_fn = train_fn
            if train_dataloader is None:
                if hasattr(DistributeExperiment, 'train_input'):
                    model_classname = getattr(Input, 'input', 0)
                    train_input_fn_class = getattr(Input, model_classname)
                    self.train_dataloader = train_input_fn_class.__new__(train_input_fn_class)
                    if self.input_mode == Input.InputOptions.TF_RECORD:
                        if features is None:
                            raise ValueError("Please peovide features for parsing the tf-record.")
                        setattr(self.train_dataloader, 'features', features)
                else:
                    raise ValueError("In Train mode, train_input_fn must be provided.")
            else:
                self.train_dataloader = train_dataloader
        elif mode == "Eval":
            if eval_fn is None:
                raise ValueError("In Eval mode, eval_fn must be provided.")
            else:
                self.eval_fn = eval_fn
            if eval_dataloader is None:
                if hasattr(DistributeExperiment, 'eval_input'):
                    model_classname = getattr(Input, 'input', 0)
                    eval_input_fn_class = getattr(Input, model_classname)
                    self.eval_dataloader = eval_input_fn_class.__new__(eval_input_fn_class)
                    if self.input_mode == Input.InputOptions.TF_RECORD:
                        if features is None:
                            raise ValueError("Please peovide features for parsing the tf-record.")
                        setattr(self.eval_dataloader, 'features', features)
                else:
                    raise ValueError("In Eval mode, eval_input_fn must be provided.")
            else:
                self.eval_dataloader = eval_dataloader
        else:
            raise ValueError("Please provide either Train or Eval as mode.")

    def train(self, pre_train_fn=None, post_train_fn=None):
        self.train_fn(self.train_dataloader, self.input_mode, pre_train_fn, post_train_fn)

    def evaluation(self, pre_eval_fn=None, post_evaluation_fn=None):
        self.eval_fn(self.eval_dataloader, pre_eval_fn, post_evaluation_fn)

    def run(self):
        if self.mode == 'Train':
            self.train()
        else:
            self.evaluation()


if __name__ == '__main__':
    pass
