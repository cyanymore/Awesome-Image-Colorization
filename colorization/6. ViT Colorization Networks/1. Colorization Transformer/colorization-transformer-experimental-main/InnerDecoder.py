# -*- coding: utf-8 -*-

from utils import *

from AttentionBlockConditional import *

class TransformerDecoderInner(nn.Module) :
  def __init__(self, D):
    super(TransformerDecoderInner, self).__init__()

    self.D = D
    self.ConditionalRowAttention = AttentionBlockConditional('row',D)

  def forward(self, input, ctx_encoder_decoder, j) :
    batch,row,col,_ = input.shape
    
    mask = torch.ones(input.shape[:-1])
    mask[:,:,j:] = -1e9
    
    out = self.ConditionalRowAttention(input, ctx_encoder_decoder, mask)

    return out

class InnerDecoder(nn.Module) :
  """
  Generate a row, one pixel at a time
  """
  def __init__(self, D):
    super(InnerDecoder, self).__init__()

    self.TransformerDecoderInner_Layer1 = TransformerDecoderInner(D) 

  def forward(self, emb_x_s_c, ctx_encoder_decoder, j):
    """
    =INPUT=


    =RETURN=
    
    z = o + ShiftRight(e)
    h = MaskedRow(z)
    p(xij) = Dense(h)
    """

    out = self.TransformerDecoderInner_Layer1(emb_x_s_c, ctx_encoder_decoder, j)

    return out
