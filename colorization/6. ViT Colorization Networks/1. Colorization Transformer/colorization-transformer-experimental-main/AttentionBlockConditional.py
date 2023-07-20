# -*- coding: utf-8 -*-

from utils import *

class MLPConditional(nn.Module) :
  def __init__(self, D):
    super(MLPConditional, self).__init__()

    self.fc1 = nn.Linear(D,D)
    self.fc2 = nn.Linear(D,D)
    self.relu = nn.ReLU()

  def forward(self, x, conv1_context_h, conv2_context_h):      
    h = self.relu(self.fc2(self.fc1(x)))
    y = conv1_context_h * h + conv2_context_h
    return y

class AttentionBlockConditional(nn.Module) :
    def __init__(self, t, D):
      """
      =INPUT=
      t : Type de Block (row/col)
           : String
      D : dimension embedding
      """
      super(AttentionBlockConditional, self).__init__()

      self.D = D
      self.Type = t

      # head 1
      self.Q1 = nn.Linear(D,D,bias=False)
      self.K1 = nn.Linear(D,D,bias=False)
      self.V1 = nn.Linear(D,D,bias=False)
      # head 2
      self.Q2 = nn.Linear(D,D,bias=False)
      self.K2 = nn.Linear(D,D,bias=False)
      self.V2 = nn.Linear(D,D,bias=False)
      # head 3
      self.Q3 = nn.Linear(D,D,bias=False)
      self.K3 = nn.Linear(D,D,bias=False)
      self.V3 = nn.Linear(D,D,bias=False)
      # head 4
      self.Q4 = nn.Linear(D,D,bias=False)
      self.K4 = nn.Linear(D,D,bias=False)
      self.V4 = nn.Linear(D,D,bias=False)
      # linear 
      self.out = nn.Linear(4*D,D)

      self.conv1_z = nn.Conv2d(D,D,kernel_size=1,bias=False)
      self.conv2_z = nn.Conv2d(D,D,kernel_size=1,bias=False)
      self.conv1_h = nn.Conv2d(D,D,kernel_size=1,bias=False)
      self.conv2_h = nn.Conv2d(D,D,kernel_size=1,bias=False)
      
      self.LN = nn.LayerNorm(D)
      self.mean_avg_pool = nn.AvgPool1d(kernel_size=1)

      self.mlp = MLPConditional(D)

    def forward(self, input, context, mask = None):
      conv1_context_z = self.conv1_z(context.transpose(1,-1)).transpose(1,-1)
      conv2_context_z = self.conv2_z(context.transpose(1,-1)).transpose(1,-1)
      
      conv1_context_h = self.conv1_h(context.transpose(1,-1)).transpose(1,-1)
      conv2_context_h = self.conv2_h(context.transpose(1,-1)).transpose(1,-1)

      batch, row, col, _ = input.shape
      out = torch.empty_like(input)
      
      s = nn.Softmax(-1)

      if self.Type == 'row' :
        for i in range(row) :   
            
          ln_input = self.LN(input[:,i,:,:])
          maski = 1 if mask is None else mask[:,i,:].unsqueeze(-1) @ torch.ones(1, col)
          
          Q1c = self.Q1(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]
          K1c = self.K1(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]
          V1c = self.V1(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]

          Q2c = self.Q2(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]
          K2c = self.K2(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]
          V2c = self.V2(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]

          Q3c = self.Q3(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]
          K3c = self.K3(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]
          V3c = self.V3(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]

          Q4c = self.Q4(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]
          K4c = self.K4(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]
          V4c = self.V4(ln_input) * conv1_context_z[:,i,:,:] + conv2_context_z[:,i,:,:]

          A1 = s(torch.matmul(Q1c, K1c.transpose(1,2)) * maski / math.sqrt(self.D))
          A2 = s(torch.matmul(Q2c, K2c.transpose(1,2)) * maski / math.sqrt(self.D))
          A3 = s(torch.matmul(Q3c, K3c.transpose(1,2)) * maski / math.sqrt(self.D))
          A4 = s(torch.matmul(Q4c, K4c.transpose(1,2)) * maski / math.sqrt(self.D))
          
          # W * W

          SA1 = torch.matmul(A1,V1c)
          SA2 = torch.matmul(A2,V2c)
          SA3 = torch.matmul(A3,V3c)
          SA4 = torch.matmul(A4,V4c)

          MSA = self.out(torch.cat((SA1,SA2,SA3,SA4),2))

          tmp = MSA + input[:,i,:,:] 

          out[:,i,:,:] = self.mlp(self.LN(tmp), conv1_context_h[:,i,:,:], conv2_context_h[:,i,:,:]) + tmp # W * D
          
      # ColumnAttention
      else :
        for j in range(col) :
          ln_input = self.LN(input[:,:,j,:])
          maskj = 1 if mask is None else mask[:,:,j].unsqueeze(-1) @ torch.ones(1, row) 
          

          Q1c = self.Q1(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]
          K1c = self.K1(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]
          V1c = self.V1(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]

          Q2c = self.Q2(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]
          K2c = self.K2(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]
          V2c = self.V2(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]

          Q3c = self.Q3(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]
          K3c = self.K3(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]
          V3c = self.V3(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]

          Q4c = self.Q4(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]
          K4c = self.K4(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]
          V4c = self.V4(ln_input) * conv1_context_z[:,:,j,:] + conv2_context_z[:,:,j,:]

          A1 = s(torch.matmul(Q1c, K1c.transpose(1,2)) * maskj / math.sqrt(self.D))
          A2 = s(torch.matmul(Q2c, K2c.transpose(1,2)) * maskj / math.sqrt(self.D))
          A3 = s(torch.matmul(Q3c, K3c.transpose(1,2)) * maskj / math.sqrt(self.D))
          A4 = s(torch.matmul(Q4c, K4c.transpose(1,2)) * maskj / math.sqrt(self.D))

          # W * W

          SA1 = torch.matmul(A1,V1c)
          SA2 = torch.matmul(A2,V2c)
          SA3 = torch.matmul(A3,V3c)
          SA4 = torch.matmul(A4,V4c)

          MSA = self.out(torch.cat((SA1,SA2,SA3,SA4),2))

          tmp = MSA + input[:,:,j,:] 

          out[:,:,j,:] = self.mlp(self.LN(tmp),conv1_context_h[:,:,j,:], conv2_context_h[:,:,j,:]) + tmp # W * D
          

      return out

