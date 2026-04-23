# Implementation of "Thinking Like Transformers" (https://arxiv.org/pdf/2106.06981.pdf)
# full repo: https://github.com/tech-srl/RASP
# @yashbonde - 18.06.2021
# MIT License
#
# Why build this?
# - learning how to write languages is the best way to learn how to minimise useless shit
#   and maximise simplicity of code + was fun to code whilst in Deep Thoughts!
#
# Where can I use this?
# - See the examples, if it's not there then will add it later.
#
# Things that are different from the paper
# -
# TODO:
# - implement conditionals
# - additional operators such as `in`, `sort`, `count`

import string
import numpy as np
import torch
import einops as ein

# NOTE: This is a demo code and not meant to be a production thing
# changing the vocab will break some tests, so for the sake of your
# and my sanity don't change this.
vocab = {k:i for i,k in enumerate(string.ascii_lowercase + "$")}
ivocab = {i:k for k,i in vocab.items()}

# ---- built in
def tokens(x, bos = False):
  """
  if bos == True, then output has bos tag added

  ### Always PAD ###

  # tokens("hello") = [ 7,  4, 11, 11, 14]
  # tokens(tokens("hello")) = "hello"
  # tokens(["hello", "hello"]) = [[7,  4, 11, 11, 14], [7, 4, 11, 11, 14]]
  # tokens([[7,  4, 11, 11, 14], [7, 4, 11, 11, 14]]) = ["hello", "hello"]

  Logic Flow:
    # Case A (str only): "hello"
    # Case B (list of str): ["hello", "hello"]
    # Case C (tensor 1D): [ 7,  4, 11, 11, 14]
    # Case D (tensor 2D): [[ 7,  4, 11, 11, 14], [ 7,  4, 11, 11, 14]]

  can consume strings, lists, arrays and tensors
  """

  if isinstance(x, str):
    # Case A (str only): "hello"
    out = torch.Tensor([vocab["$"]] + [vocab[t] for t in x.lower()]).long()
    if not bos:
      out = out[1:]
    return out
  elif isinstance(x, list) and isinstance(x[0], str):
    # Case B (list of str): ["hello", "hello"]
    m = max([len(y) for y in x])
    for i,y in enumerate(x):
      x[i] = x[i] + "".join(["$" for _ in range(m - len(x[i]))])
    return torch.cat([tokens(s, bos).unsqueeze(0) for s in x]).long()
  else:
    assert isinstance(x, (torch.Tensor, np.ndarray)), "Can consume only strings and torch.Tensors / np.ndarrays"
    # input is likely a tensor
    if len(x.shape) == 1:
      # Case C (tensor 1D): [ 7,  4, 11, 11, 14]
      out = "".join([ivocab[t] for t in x.tolist()])
      # FORCE REMOVE PADDING
      if "$" in out[1:]:
        out = out[:out[1:].index("$")]
      if not bos and out[0] == "$":
        out = out[1:]
      out = out
      return out
    else:
      # Case D (tensor 2D): [ [ 7,  4, 11, 11, 14], [ 7,  4, 11, 11, 14]]
      return [tokens(s, bos) for s in x]

def indices(x):
  # indices("hello") = [0,1,2,3,4]
  pass

def length(x):
  # length("hello") = [5,5,5,5,5]
  pass


# --- element wise
def logical(x, op, y = None):
  # logical(x, "and", y)
  pass

def elementwise(x, op, y):
  # elementwise(x, "-", y)
  pass


# --- select
def select(m1: torch.Tensor, m2, op):
  # creating boolean matrices called "selectors"
  pass
  
# --- aggregate
def aggregate(s, x, agg = "mean"):
  # collapsing selectors and s-ops into new s-ops
  pass

# --- simple select aggregate
def flip(x):
  pass

# --- selector_width

# def selector_width(x):
#   pass

# def selector_width (sel ,assume_bos = False):
#   light0 = indicator ( indices == 0)
#   or0 = sel or select_eq ( indices ,0)
#   and0 =sel and select_eq ( indices ,0)
#   or0_0_frac =aggregate (or0 , light0 )
#   or0_width = 1 / or0_0_frac
#   and0_width = aggregate (and0 ,light0 ,0)
# 
#   # if has bos , remove bos from width
#   # (doesn ’t count , even if chosen by
#   # sel) and return .
#   bos_res = or0_width - 1
# 
#   # else , remove 0 - position from or0 ,
#   # and re -add according to and0 :
#   nobos_res = bos_res + and0_width
# 
#   return bos_res if assume_bos else
#   nobos_res