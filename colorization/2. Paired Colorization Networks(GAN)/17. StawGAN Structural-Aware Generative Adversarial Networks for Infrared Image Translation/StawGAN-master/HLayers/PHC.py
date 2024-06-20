from   numpy.random            import RandomState
from scipy.stats import chi
import torch 
import numpy as np
from torch import nn
import torch.nn.functional as F

import math
seed = 888
device = "cuda" if torch.cuda.is_available() else "cpu"

def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1,1234))

    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4,loc=0,scale=s,size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0,1.0,number_of_weights)
    v_j = np.random.uniform(-1.0,1.0,number_of_weights)
    v_k = np.random.uniform(-1.0,1.0,number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 +0.0001)
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)
    weight_k = modulus * v_k*np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)


def get_weight(n,in_f,out_f,kernel_size,criterion):
    r, i, j, k = quaternion_init(
        in_f,
        out_f//n,
        rng=RandomState(777),
        kernel_size=kernel_size,
        criterion=criterion
    )
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    if criterion=="he":
        return torch.cat([r,i,j,k],dim=1)
    else:
        return torch.cat([r.squeeze(1),i.squeeze(1),j.squeeze(1),k.squeeze(1)],dim=0)

def get_s_init(kernel_size,in_f,out,criterion):
    if criterion == "glorot":
        w_shape = (out, in_f) + (*kernel_size,)
        r_weight = torch.Tensor(*w_shape)
        i_weight = torch.Tensor(*w_shape)
        j_weight = torch.Tensor(*w_shape)
        k_weight = torch.Tensor(*w_shape)
    else:
        r_weight = torch.Tensor(in_f,out)
        i_weight = torch.Tensor(in_f,out)
        j_weight = torch.Tensor(in_f,out)
        k_weight = torch.Tensor(in_f,out)
    # print(torch.stack([r_weight,i_weight,j_weight,k_weight],dim=2).shape)

    r, i, j, k = quaternion_init(
        r_weight.size(1),
        r_weight.size(0),
        rng=RandomState(seed),
        kernel_size=kernel_size,
        criterion=criterion
    )

    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)
    
    return torch.stack([r_weight,i_weight,j_weight,k_weight],dim=0).to(device)

def get_a_init(n,k_size,criterion):
    r_weight = torch.Tensor(n,n,n)
    i_weight = torch.Tensor(n,n,n)
    j_weight = torch.Tensor(n,n,n)
    k_weight = torch.Tensor(n,n,n)

    r, i, j, k = quaternion_init(
        r_weight.size(1),
        r_weight.size(0),
        rng=RandomState(seed),
        kernel_size=(n,n),
        criterion=criterion
    )

    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)
    return torch.stack([r_weight,i_weight,j_weight,k_weight],dim=0)[:,0,0].to(device)
########################
## STANDARD PHM LAYER ##
########################

class PHMLinear(nn.Module):

  def __init__(self, n, in_features, out_features,bias=True):
    super(PHMLinear, self).__init__()
    self.n = n if ((in_features //n>0) and (out_features //n>0) ) else 1
    self.in_features = in_features
    self.out_features = out_features

    #self.a = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((self.n, self.n, self.n))))
    # self.s = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((self.n, self.out_features//self.n, self.in_features//self.n))))
    #self.a = get_a_init(self.n,None,"he")
    mat1 = torch.eye(4).view(1, 4, 4)

    # Define the four matrices that summed up build the Hamilton product rule.
    mat2 = torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]]).view(1, 4, 4)
    mat3 = torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]]).view(1, 4, 4)
    mat4 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]]).view(1, 4, 4)
    self.a=nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    self.s = nn.Parameter(get_s_init(None, self.in_features//self.n,self.out_features//self.n,"he"))

    self.old_weight = torch.zeros((self.out_features, self.in_features))
    self.weight = get_weight(self.n,self.out_features,self.in_features,None,"he")
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if bias:
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.uniform_(self.bias, -bound, bound)
    else:
        self.register_parameter('bias', None)

    #self.reset_parameters()
  def kronecker_product1(self, a, b): #adapted from Bayer Research's implementation
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features))
    for i in range(self.n):
        H = H + torch.kron(self.a[i], self.s[i])
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
    #self.weight = self.kronecker_product2()
    input = input.type(dtype=self.weight.type())
    return F.linear(input, weight=self.weight, bias=self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    mat1 = torch.eye(4).view(1, 4, 4)

    # Define the four matrices that summed up build the Hamilton product rule.
    mat2 = torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]]).view(1, 4, 4)
    mat3 = torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]]).view(1, 4, 4)
    mat4 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]]).view(1, 4, 4)
    self.a=nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    self.s = nn.Parameter(get_s_init(None, self.in_features//self.n,self.out_features//self.n,"he"))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if self.bias is not None:
        nn.init.uniform_(self.bias, -bound, bound)

#############################
## CONVOLUTIONAL PHM LAYER ##
#############################

class PHMConv(nn.Module):

  def __init__(self, n, in_features, out_features, kernel_size, stride=1, padding=0,bias = True):
    super(PHMConv, self).__init__()
    self.n = n #if ((in_features //n>0) and (out_features //n>0) ) else 1
    self.in_features = in_features# if in_features //n>0 else n
    self.out_features = out_features# if out_features //n>0 else n
    
    self.padding = padding
    self.stride = stride
    #self.a = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((self.n, self.n, self.n))))
    # self.s = nn.Parameter(torch.nn.init.xavier_uniform_(
    #     torch.zeros((self.n, self.out_features//self.n, self.in_features//self.n, kernel_size, kernel_size))))
    self.kernel_size = kernel_size
    #self.a = nn.Parameter(get_a_init(self.n,(kernel_size,kernel_size),"glorot"))
    mat1 = torch.eye(4).view(1, 4, 4)

    # Define the four matrices that summed up build the Hamilton product rule.
    mat2 = torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]]).view(1, 4, 4)
    mat3 = torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]]).view(1, 4, 4)
    mat4 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]]).view(1, 4, 4)
    self.a=nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    self.s = nn.Parameter(get_s_init((kernel_size,kernel_size),self.in_features//self.n,self.out_features//self.n,"glorot"))
    
    self.weight = torch.zeros((self.out_features, self.in_features))
    #self.weight = get_weight(self.n,1,self.out_features,self.kernel_size,"glorot")
    
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if bias:
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.uniform_(self.bias, -bound, bound)
    else:
        self.register_parameter('bias', None)
    #self.reset_parameters()

  def kronecker_product1(self, a, s):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(s.shape[-4:-2]))
    siz2 = torch.Size(torch.tensor(s.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * s.unsqueeze(-4).unsqueeze(-6)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size)).to(device)
    for i in range(self.n):
        kron_prod = torch.kron(self.a[i], self.s[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size).to(device)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
    # self.weight = self.kronecker_product2()
    input = input.type(dtype=self.weight.type())
        
    return F.conv2d(input, weight=self.weight, stride=self.stride, padding=self.padding,bias=self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    # init.kaiming_uniform_(self.a, a=math.sqrt(5))
    # init.kaiming_uniform_(self.s, a=math.sqrt(5))
    mat1 = torch.eye(4).view(1, 4, 4)

    # Define the four matrices that summed up build the Hamilton product rule.
    mat2 = torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]]).view(1, 4, 4)
    mat3 = torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]]).view(1, 4, 4)
    mat4 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]]).view(1, 4, 4)
    self.a=nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    self.s = nn.Parameter(get_s_init((self.kernel_size, self.kernel_size),self.in_features//self.n,self.out_features//self.n,"glorot"))

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if self.bias is not None:
        nn.init.uniform_(self.bias, -bound, bound)


class PHMTransposeConv(nn.Module):

  def __init__(self, n, in_features, out_features, kernel_size, stride=1, padding=0,bias = True,output_padding=0):
    super(PHMTransposeConv, self).__init__()
    self.n = n #if ((in_features //n>0) and (out_features //n>0) ) else 1
    self.in_features = in_features# if in_features //n>0 else n
    self.out_features = out_features# if out_features //n>0 else n
    
    self.padding = padding
    self.output_padding = output_padding
    self.stride = stride
    #self.a = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((self.n, self.n, self.n))))
    # self.s = nn.Parameter(torch.nn.init.xavier_uniform_(
    #     torch.zeros((self.n, self.out_features//self.n, self.in_features//self.n, kernel_size, kernel_size))))
    self.kernel_size = kernel_size
    #self.a = nn.Parameter(get_a_init(self.n,(kernel_size,kernel_size),"glorot"))
    mat1 = torch.eye(4).view(1, 4, 4)

    # Define the four matrices that summed up build the Hamilton product rule.
    mat2 = torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]]).view(1, 4, 4)
    mat3 = torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]]).view(1, 4, 4)
    mat4 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]]).view(1, 4, 4)
    self.a = nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    self.s = nn.Parameter(get_s_init((kernel_size,kernel_size),self.out_features//self.n,self.in_features//self.n,"glorot"))
    
    self.weight = torch.zeros((self.out_features, self.in_features))
    #self.weight = get_weight(self.n,1,self.out_features,self.kernel_size,"glorot")
    
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if bias:
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.uniform_(self.bias, -bound, bound)
    else:
        self.register_parameter('bias', None)
    #self.reset_parameters()

  def kronecker_product1(self, a, s):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(s.shape[-4:-2]))
    siz2 = torch.Size(torch.tensor(s.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * s.unsqueeze(-4).unsqueeze(-6)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size)).to(device)
    for i in range(self.n):
        kron_prod = torch.kron(self.a[i], self.s[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size).to(device)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
    # self.weight = self.kronecker_product2()
    input = input.type(dtype=self.weight.type())
        
    return F.conv_transpose2d(input, weight=self.weight, stride=self.stride, padding=self.padding, output_padding = self.output_padding, bias=self.bias)


  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    # init.kaiming_uniform_(self.a, a=math.sqrt(5))
    # init.kaiming_uniform_(self.s, a=math.sqrt(5))
    mat1 = torch.eye(4).view(1, 4, 4)

    # Define the four matrices that summed up build the Hamilton product rule.
    mat2 = torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]]).view(1, 4, 4)
    mat3 = torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]]).view(1, 4, 4)
    mat4 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]]).view(1, 4, 4)
    self.a=nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    self.s = nn.Parameter(get_s_init((self.kernel_size, self.kernel_size),self.in_features//self.n,self.out_features//self.n,"glorot"))

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if self.bias is not None:
        nn.init.uniform_(self.bias, -bound, bound)