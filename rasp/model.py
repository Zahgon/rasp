# give proper credits, stole from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# but he won't mind!
# modified for yashbonde/rasp

import math
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import einops as ein

from rasp.manual import vocab, tokens

# ------ configurations ------ #

class Config:
  def __init__(self, **kwargs):
    self.vocab_size = len(vocab)
    self.n_embd = 18
    self.block_size = 32
    self.dropout = 0.0
    self.n_layer = 1
    self.n_head = 1
    for k,v in kwargs.items():
      setattr(self, k, v)

  def get_json(self):
    pass

# ------ response ------ #

class Response:
  def __init__(self, logits, loss, attns):
    self.logits = logits
    self.loss = loss
    self.attns = attns
    self.tokens = tokens(logits.argmax(-1))

# ------ model ------ #

class SelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # single qkv like GPT
    self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
    self.split_size = config.n_embd
    
    # output projection
    self.proj = nn.Linear(config.n_embd, config.n_embd)
    
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                  .view(1, 1, config.block_size, config.block_size))
    self.n_head = config.n_head

  def forward(self, x, attn_mask = None):
    pass


class Block(nn.Module):
  """ an unassuming Transformer block """

  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.split_size = config.n_embd
    self.attn = SelfAttention(config)
    self.mlp = nn.Sequential(
      nn.Linear(config.n_embd, 4 * config.n_embd),
      nn.GELU(),
      nn.Linear(4 * config.n_embd, config.n_embd),
      nn.Dropout(config.dropout),
    )

    self.n_head = config.n_head

  def forward(self, x):
    pass


class FullTransformer(nn.Module):
  """A full transformer model with focus on data and I/O.
  Can consume strings, lists, arrays and torch-tensors."""

  def __init__(self, config):
    super().__init__()

    # input embedding stem
    self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
    self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
    self.drop = nn.Dropout(config.dropout)
    
    # transformer
    self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
    
    # decoder head
    self.ln_f = nn.LayerNorm(config.n_embd)
    self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.block_size = config.block_size

    self.config = config

  @property
  def num_parameters(self):
    pass

  def get_device(self):
    pass
  
  def format_inputs_and_tokens(self, idx, targets):
    pass

  def forward(self, idx, targets=None, output_dict = False):
    """
    Args:
      idx: Can take in following objects:
        - string
        - list[string]
        - torch.LongTensor (1D)
        - torch.LongTensor (2D)
      targets (optional): Since in rasp you calculate losses for attention matrix
        as well, this targets is a list:
          - torch.LongTensor(): with the cross entropy for entire input tokens,
            just like a normal transformer (GPT/BERT)
          - target_attn_masks:  this is the target matrices for all the attentions in the network.
            ensure that the number of heads and values are common.

    Returns:
        [type]: [description]
    """
    pass

# ------ model function ------ #

def get_model(**kwargs):
  config = Config(**kwargs)
  return FullTransformer(config)
