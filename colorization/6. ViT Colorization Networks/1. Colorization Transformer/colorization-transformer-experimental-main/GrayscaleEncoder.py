# -*- coding: utf-8 -*-

from utils import *
from AttentionBlock import *

class TransformerEncoder(nn.Module) :
  def __init__(self, D):
    super(TransformerEncoder, self).__init__()

    self.D = D
    self.RowAttention = AttentionBlock('row',D)
    self.ColumnAttention = AttentionBlock('col',D)

  def forward(self, input) :
    input = self.RowAttention(input)
    input = self.ColumnAttention(input)
    return input

class GrayscaleEncoder(nn.Module) :
  """
  The encoder encodes each prior channel independently with a stack of unmasked
  row/column attention layers.
  """

  def __init__(self, D):
    super(GrayscaleEncoder, self).__init__()

    self.D = D

    self.TransformerEncoder_Layer1 = TransformerEncoder(D) 
    #self.TransformerEncoder_Layer2 = TransformerEncoder(D)
    #self.TransformerEncoder_Layer3 = TransformerEncoder(D)
    #self.TransformerEncoder_Layer4 = TransformerEncoder(D)


  def forward(self, embedding_x_g):
    """
    =INPUT=
    embedding_x_g : Embedding de la channel courante
                  : (BATCH *) M * N * D

    =RETURN=
    out : Context de la channel courante
        : (BATCH *) M * N * D

    """
    out = self.TransformerEncoder_Layer1(embedding_x_g)
    #out = self.TransformerEncoder_Layer2(out)
    #out = self.TransformerEncoder_Layer3(out)
    #out = self.TransformerEncoder_Layer4(out)

    return out
