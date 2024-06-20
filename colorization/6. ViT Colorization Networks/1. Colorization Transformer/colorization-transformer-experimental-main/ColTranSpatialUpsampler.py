# -*- coding: utf-8 -*-

from utils import *
from GrayscaleEncoder import *

class ColTranSpatialUpsampler(nn.Module) :
  """ 
  AxialTransformer
  """

  def __init__(self, D, NColor, H, W):
    super(ColTranSpatialUpsampler, self).__init__()

    self.D = D
    self.H = H
    self.W = W
    self.NColor = NColor

    self.embedding_x_g = nn.Embedding(NColor,D) # BATCH * H * W * D
    self.embedding_x_rgb = nn.Sequential(nn.Embedding(NColor,D), nn.Embedding(NColor,D), nn.Embedding(NColor,D))

    self.grayscale_encoder = GrayscaleEncoder(D)

    # 256 nuances de couleurs
    self.linear = nn.Linear(D,NColor)


  def forward(self, x_g, x_s) :
    """
    =INPUT=
    x_s : low resolution image, a spatially downsampled representation of x
        : M * N * 3
    x_g : 
             : H * W * 1

    =RETURN=
    x : colorized image (high resolution)
      : H * W * 3
    """
    
    
    x_s = F.interpolate(x_s.permute(0,3,1,2).float(),size=(self.H, self.W),mode="bilinear").permute(0,2,3,1).long()
    
    batch,row,col,channel = x_s.shape

    pe = positionalencoding2d(self.D, row, col, batch)
    
    emb_g = pe + self.embedding_x_g(x_g)  
    
    out = torch.zeros(batch, row, col, channel, self.NColor)   
    

    for k in range(channel):
        
        emb_k = pe + self.embedding_x_rgb[k](x_s[:,:,:,k])  

        input_encoder = emb_g + emb_k

        out_encoder = self.grayscale_encoder(input_encoder)
  
        out[:,:,:,k] = self.linear(out_encoder)

    return out.argmax(-1), out
