# Since building simple primitives is the primary task of this language,
# training the model should have first class support.
# now you can directly load a primitive as follows:
#
# >>> from rasp import Primitive
# >>> reverse = Primitive("reverse")
# >>> reverse("hey")
# ... "yeh"

import json
import numpy as np
from tqdm import trange

from rasp.model import *
from rasp.manual import ivocab, vocab, tokens
from rasp.daily import Hashlib

def set_seed(seed):
  pass


class Primitive:
  # primtive class is a Transformer neural network whose objective
  # is to perform that particular task.
  def __init__(self, name, code = None, seed = 4, **model_kwargs):
    set_seed(4)
    if code is not None:
      raise NotImplementedError("code parsing is still not implemented, hold your horses!")

    self.model = get_model(**model_kwargs)
    self.name = name

    str_ = f"{name}-" + json.dumps(self.model.config.get_json())
    self._hash = Hashlib.sha256(str_)

  def get_parameters(self):
    pass

  def __call__(self, *args, **kwargs):
    return self.model(*args, **kwargs)

  def viz(self, x):
    # this is not the best visualisation of attention since the values are
    # in float. But this is good enough to see what's up
    pass

  def train(self, ds, man_fn, optim_name = "Adam", n_epochs = 5, pbar = False, **optimiser_params):
    """training any primitive has first class support since this is what each primitive is"""
    pass

def get_vocab():
  return vocab, ivocab
