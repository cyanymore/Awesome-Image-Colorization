# -*- coding: utf-8 -*-

from utils import *
from GrayscaleEncoder import *
from OuterDecoder import *
from InnerDecoder import *

class ColTranCore(nn.Module) :
  """ 
  AxialTransformer
  """

  def __init__(self, D, nb_colors):
    super(ColTranCore, self).__init__()

    self.D = D
    self.nb_colors = nb_colors

    self.embedding_x_g = nn.Embedding(nb_colors, D) # H * W * D
    self.embedding_x_s_c = nn.Embedding(nb_colors, D) # H * W * D

    self.grayscale_encoder = GrayscaleEncoder(D)
    self.outer_decoder = OuterDecoder(D)
    self.inner_decoder = InnerDecoder(D)

    self.out_inner_decoder = nn.Linear(D,512)
    self.out_grayscale_encoder = nn.Linear(D,512)
    
  def sampling(self, proba):
        valeur = torch.linspace(0, self.nb_colors, 8, dtype=torch.long)
        b, m, n, l = proba.shape
        x_hat_s_c = torch.zeros(b,m,n,3)
        
        for i in range(m):
            for j in range(n):
                prob_dist = torch.distributions.Categorical(proba[:,i,j,:])
                rgb = intTo3bit(prob_dist.sample()) 
                for k in range(3):
                    x_hat_s_c[:,i,j,k] = valeur[rgb[k]]
        return x_hat_s_c

  def forward(self, x_g, x_s_c = None) :
    """
    =INPUT=
    x_g : B * M * N
    x_s_c : B * M * N * 3

    =RETURN=
    proba : B * M * N * 8^3
    """
    
    
    batch, row, col, channel = x_g.shape if x_s_c is None else x_s_c.shape
    
    pe = positionalencoding2d(self.D, row, col, batch)

    out_g = pe + self.embedding_x_g(x_g)
    out_i = torch.zeros(batch, row, col, self.D)
    out_o = torch.zeros(batch, row, col, self.D)
    
    projection = torch.zeros(batch, row, col, 512)
    
    x_hat_s_c = torch.zeros(batch, row, col, 3)

    
    for k in range(channel):
        x_s_ck = self.embedding_x_s_c((x_g if x_s_c is None else x_s_c)[:,:,:,k]) + pe
      
        out_g = self.grayscale_encoder(out_g)
        
        for i in range(row):
            out_o = self.outer_decoder(x_s_ck, out_g, i)
            
            context_i = (out_g + out_o)[:,i].unsqueeze(1)
            input_i = (context_i + shift_right(x_s_ck))[:,i].unsqueeze(1)
            
            for j in range(col):
                
                out_i = self.inner_decoder(input_i, context_i, j)
                
                projection[:,i,j] += self.out_inner_decoder(out_i[:,0,j])
                # B * M * N * 8**3           
        
        # On reset conditionnal row attention               
        self.outer_decoder = OuterDecoder(self.D)

        
    x_hat_s_c = self.sampling(projection.softmax(-1))    

    return x_hat_s_c, projection, self.out_grayscale_encoder(out_g) 
