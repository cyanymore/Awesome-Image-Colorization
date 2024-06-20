# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utils import *

from ColTranCore import *
from ColTranColorUpsampler import *
from ColTranSpatialUpsampler import *

#@title Constantes utiles

M = N = 64
#D = 512 # multiple de 4 : dimension embedding
D = 32
H = W = 256 

root256='./dataset/tiny_imagenet_256'
root256g='./dataset/tiny_imagenet_256_g'
root64='./dataset/tiny_imagenet_64'
root64c='./dataset/tiny_imagenet_64_c'
root64g='./dataset/tiny_imagenet_64_g'

#NB_EPOCH_CORE = 450000
#NB_EPOCH_COLOR = 300000
#NB_EPOCH_SPATIAL = 150000

#BATCH_SIZE_CORE = 224
#BATCH_SIZE_COLOR = 768
#BATCH_SIZE_SPATIAL = 32

NB_EPOCH_CORE = 20
NB_EPOCH_COLOR = 20
NB_EPOCH_SPATIAL = 20

BATCH_SIZE_CORE = 64
BATCH_SIZE_COLOR = 128
BATCH_SIZE_SPATIAL = 32

criterion = nn.CrossEntropyLoss()

core = ColTranCore(D, 256)
color = ColTranColorUpsampler(D, 256)
spatial = ColTranSpatialUpsampler(D, 256, 8, 8)
      

train_core = True
###############################################################################
"""Apprentissage ColTran Core"""

if train_core:
    print("##############")
    print("TRAINING CORE")
    print("##############")
    
    core.train()
    
    optim = torch.optim.RMSprop(core.parameters(), 3e-4)
    
    """ REAL """
    #train_dataset = ColTranDataset(root64g, root64c)
    #train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE_CORE,drop_last=True,shuffle=True,pin_memory=True) 
    
    """ TOYS """
    train_dataset_toy = ColTranCoreDatasetToy(1000)
    train_loader = DataLoader(train_dataset_toy,batch_size=BATCH_SIZE_CORE,drop_last=True,shuffle=True) 
    
    loss = []
    
    for epoch in tqdm(range(NB_EPOCH_CORE)):
        loss_train = 0        
        for x_g, x_s_c in train_loader:

            optim.zero_grad()
            
            x_s_c_hat, projection, projection_tilt = core(x_g, x_s_c)
            
            #valeur entre 0 et 511
            x_one_channel = toOneChannel(x_s_c).long()
            l = criterion(projection.permute(0,3,1,2), x_one_channel)
            loss_train += l.item()
            
            l.backward()
            optim.step()
            
        print("loss => ",loss_train/BATCH_SIZE_CORE)
        loss.append(loss_train/BATCH_SIZE_CORE)
    
    draw_result(range(20), loss, "Loss Cross-entropy [ColTran CORE]")
            
            
train_color = False
###############################################################################
"""Apprentissage ColTran Color Upsampler"""

if train_color:
    
    print("##############")
    print("TRAINING COLOR UPSAMPLER")
    print("##############")
    
    color.train()
    
    """ REAL """
    #train_dataset = ColTranDataset(root64g, root64c, root64)
    #train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE_COLOR,drop_last=True,shuffle=True) 
    
    """ TOYS """
    train_dataset_toy = ColTranUpColorDatasetToy(1000)
    train_loader = DataLoader(train_dataset_toy,batch_size=BATCH_SIZE_COLOR,drop_last=True,shuffle=True) 
    
    optim = torch.optim.RMSprop(color.parameters(), 3e-4)
    
    loss = []
    
    for epoch in tqdm(range(NB_EPOCH_COLOR)):
        loss_train = 0 
        for x_g, x_s_c, x_s in train_loader:
            optim.zero_grad()
            
            x_s_hat, projection = color(x_g, x_s_c)
            
            l = criterion(projection.permute(0,4,1,2,3), x_s)
            loss_train += l.item()
            
            l.backward()
            optim.step() 
            
        print("loss => ",loss_train/BATCH_SIZE_COLOR)
        loss.append(loss_train/BATCH_SIZE_COLOR)
    
    draw_result(range(20), loss, "Loss Cross-entropy [ColTran UPSAMPLER COLOR]")
            
train_spatial = False
###############################################################################
"""Apprentissage ColTran Spatial Upsampler"""

if train_spatial:
    
    print("##############")
    print("TRAINING SPATIAL")
    print("##############")
    
    spatial.train()
    
    """ REAL """
    #train_dataset = ColTranDataset(root256g, root64, root256)
    # 256*256 // 64*64*3 // 256*256*3
    #train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE_SPATIAL,drop_last=True,shuffle=True,pin_memory=True) 
    
    """ TOYS """
    train_dataset_toy = ColTranUpSpaceDatasetToy(1000)
    train_loader = DataLoader(train_dataset_toy,batch_size=BATCH_SIZE_SPATIAL,drop_last=True,shuffle=True) 
    
    optim = torch.optim.RMSprop(spatial.parameters(), 3e-4)
    
    loss = []
    
    for epoch in tqdm(range(NB_EPOCH_SPATIAL)):
        loss_train = 0
        for x_g, x_s, x in train_loader:
            optim.zero_grad()
            
            x_hat, projection = spatial(x_g, x_s)
            
            l = criterion(projection.permute(0,4,1,2,3), x)
            loss_train += l.item()
            
            l.backward()
            optim.step()
            
        print("loss => ",loss_train/BATCH_SIZE_SPATIAL)
        loss.append(loss_train/BATCH_SIZE_SPATIAL)
        
    draw_result(range(20), loss, "Loss Cross-entropy [ColTran UPSAMPLER SPATIAL]")
        
        
eval = True
###############################################################################
"""Evaluation"""

if eval:
    print("##############")
    print("TESTING")
    print("##############")
    
    with torch.no_grad() :
        core.eval()
        color.eval()
        spatial.eval()

        """ TOYS """
        test_dataset_toy = ColTranDatasetToy(3)
        test_loader = DataLoader(test_dataset_toy,batch_size=1,drop_last=True,shuffle=True)
        
        i = 0
        
        for x_g_s, x_g, x_s_c in test_loader:
            
            i_xgs = Image.fromarray(x_g_s[0].numpy().astype(np.uint8))
            i_xgs.save("./test/xg_"+str(i)+".png")
                
            x_s_c_hat, projection, projection_tilt = core(x_g_s, x_s_c)
            x_s, projection = color(x_g_s, x_s_c)
            x, projection = spatial(x_g, x_s)
            
            i_xg = Image.fromarray(x[0].numpy().astype(np.uint8),'RGB')
            i_xg.save("./test/xg_pred_"+str(i)+".png")
            
            i+=1
            


