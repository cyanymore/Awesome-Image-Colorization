# -*- coding: utf-8 -*-

from utils import *
from AttentionBlockConditional import *


class TransformerDecoderOuter(nn.Module) :
  def __init__(self, D):
    super(TransformerDecoderOuter, self).__init__()

    self.D = D
    self.ConditionalRowAttention = AttentionBlockConditional('row',D)
    self.ConditionalColumnAttention = AttentionBlockConditional('col',D)
    self.row = None

  def forward(self, input, ctx_grayscale_encoder, i) :
    batch,row,col,_ = input.shape
    
    mask = torch.ones(input.shape[:-1])
    mask[:,i+1:,:] = -1e9
    
    if self.row is None:
        self.row = self.ConditionalRowAttention(input, ctx_grayscale_encoder)

    out = self.ConditionalColumnAttention(self.row, ctx_grayscale_encoder, mask)

    return out

class OuterDecoder(nn.Module) :
  def __init__(self, D):
    super(OuterDecoder, self).__init__()
    self.D = D
    self.TransformerDecoderOuter_Layer1 = TransformerDecoderOuter(D) 

  def forward(self, emb_x_s_c, ctx_grayscale_encoder, i):
    """
        |e = Embeddings(x)
    N x |s_o = MaskedColumn(Row(e))
        |o = ShiftDown(s_o)
        
    """
    out = self.TransformerDecoderOuter_Layer1(emb_x_s_c, ctx_grayscale_encoder, i)
    
    out = shift_down(out)

    return out
