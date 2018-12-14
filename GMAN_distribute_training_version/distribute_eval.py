#  ====================================================
#   Filename: distribute_eval.py
#   Author: Botao Xiao
#   Function: The training file is used to save the eval process
#  ====================================================


class Eval(object):
    def __init__(self, data_loader, input_mode):
        self.data_loader = data_loader
        self.input_mode = input_mode

    def eval(self,
             pre_eval_fn=None,
             post_eval_fn=None,
             pre_process_fn=None,
             post_process_fn=None,
             *args,
             **kwargs):
        pass

    def run(self):
        self.eval()