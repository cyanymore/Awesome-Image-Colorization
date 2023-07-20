# -*- coding: utf-8 -*-

from utils import *
from GrayscaleEncoder import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ColTranColorUpsampler(nn.Module) :
  """ 
  AxialTransformer
  """

  def __init__(self, D, NColor):
    super(ColTranColorUpsampler, self).__init__()

    self.D = D
    self.NColor = NColor

    self.embedding_x_g = nn.Embedding(NColor,D) # BATCH * M * N * D
    self.embedding_x_rgb = nn.Sequential(nn.Embedding(NColor,D), nn.Embedding(NColor,D), nn.Embedding(NColor,D))

    self.grayscale_encoder = GrayscaleEncoder(D)

    # 256 nuances de couleurs
    self.linear = nn.Linear(D,NColor)


  def forward(self, x_g, x_s_c) :
    """
    =INPUT=
    x_s_c : low resolution image, a spatially downsampled representation of x
        : M * N * 3
    x_g : 
             : M * N * 1

    =RETURN=
    x : colorized image (high resolution)
      : H * W * 3
    """
    
    batch,row,col,channel = x_s_c.shape
    
    pe = positionalencoding2d(self.D, row, col, batch)
    
    emb_g = pe + self.embedding_x_g(x_g)  
    
    out = torch.zeros(batch, row, col, channel, self.NColor)    
    

    for k in range(channel):
        
        emb_k = pe + self.embedding_x_rgb[k](x_s_c[:,:,:,k])       

        input_encoder = emb_g + emb_k

        out_encoder = self.grayscale_encoder(input_encoder)
  
        out[:,:,:,k] = self.linear(out_encoder)

    return out.argmax(-1), out
