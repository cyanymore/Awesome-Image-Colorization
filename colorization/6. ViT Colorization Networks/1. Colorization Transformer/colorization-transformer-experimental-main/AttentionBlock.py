# -*- coding: utf-8 -*-

from utils import *

class MLP(nn.Module) :
  def __init__(self, D):
    super(MLP, self).__init__()

    self.fc1 = nn.Linear(D,D)
    self.fc2 = nn.Linear(D,D)
    self.relu = nn.ReLU()

  def forward(self,x) :
    return self.relu(self.fc2(self.fc1(x)))

class AttentionBlock(nn.Module) :
    def __init__(self, t, D):
      """
      =INPUT=
      type : Type de Block (row/col)
           : String
      D : dimension embedding
      """
      super(AttentionBlock, self).__init__()

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

      self.LN = nn.LayerNorm(D)

      self.mlp = MLP(D)

    def forward(self, input):
      """
      =INPUT=
      type : (BATCH *) M * N * D
      
      =RETURN=
      out : (BATCH *) M * N * D
  
      """
      batch, row, col, _ = input.shape
      out = torch.empty_like(input)
      
      s = torch.nn.Softmax(-1)

      if self.Type == 'row' :          
        for i in range(row) :
          ln_input = self.LN(input[:,i,:,:])

          A1 = s(torch.matmul(self.Q1(ln_input),
                                      self.K1(ln_input).transpose(1,2))/math.sqrt(self.D))
          A2 = s(torch.matmul(self.Q2(ln_input),
                                      self.K2(ln_input).transpose(1,2))/math.sqrt(self.D))
          A3 = s(torch.matmul(self.Q3(ln_input),
                                      self.K3(ln_input).transpose(1,2))/math.sqrt(self.D))
          A4 = s(torch.matmul(self.Q4(ln_input),
                                      self.K4(ln_input).transpose(1,2))/math.sqrt(self.D))
          # A : BATCH * W * W

          SA1 = torch.matmul(A1,self.V1(ln_input))
          SA2 = torch.matmul(A2,self.V2(ln_input))
          SA3 = torch.matmul(A3,self.V3(ln_input))
          SA4 = torch.matmul(A4,self.V4(ln_input))

          MSA = self.out(torch.cat((SA1,SA2,SA3,SA4),2))
 
          tmp = MSA + input[:,i,:,:] 
  
          out[:,i,:,:] = self.mlp(self.LN(tmp)) + tmp # W * D
          
      # ColumnAttention
      else :
        for j in range(col) :
          ln_input = self.LN(input[:,:,j,:])

          A1 = s(torch.matmul(self.Q1(ln_input),
                                      self.K1(ln_input).transpose(1,2))/math.sqrt(self.D))
          A2 = s(torch.matmul(self.Q2(ln_input),
                                      self.K2(ln_input).transpose(1,2))/math.sqrt(self.D))
          A3 = s(torch.matmul(self.Q3(ln_input),
                                      self.K3(ln_input).transpose(1,2))/math.sqrt(self.D))
          A4 = s(torch.matmul(self.Q4(ln_input),
                                      self.K4(ln_input).transpose(1,2))/math.sqrt(self.D))
          # Ai : BATCH * H * H

          SA1 = torch.matmul(A1,self.V1(ln_input))
          SA2 = torch.matmul(A2,self.V2(ln_input))
          SA3 = torch.matmul(A3,self.V3(ln_input))
          SA4 = torch.matmul(A4,self.V4(ln_input))

          MSA = self.out(torch.cat((SA1,SA2,SA3,SA4),2))
 
          tmp = MSA + input[:,:,j,:]

          out[:,:,j,:] = self.mlp(self.LN(tmp)) + tmp # H * D

      return out
