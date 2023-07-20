import torch
from torch import nn
from dataloader import DroneVeichleDatasetPreTraining
from models import Generator
from tqdm import tqdm
import utils 
from torch.utils.data import DataLoader

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = 0
        for data in tqdm(train_dl):
            ir, rgb, lab = data[0].to(device), data[1].to(device), data[2].to(device)
            c_trg = utils.label2onehot(lab, 2).to(device)
            #print(ir.shape, rgb.shape)
            preds = net_G(ir, c=c_trg, mode="pretrain")
            loss = criterion(preds, rgb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter+=loss.item()
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter/len(train_dl):.5f}")

device = "cuda" if torch.cuda.is_available() else "cpu"
in_c = 3
net_G = Generator(in_c = in_c + 2,
                  mid_c = 64,
                  layers = 2,
                  s_layers=3,
                  affine=True,
                  last_ac=True,
                  colored_input=True).to(device)
opt = torch.optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()
data = DroneVeichleDatasetPreTraining(split="train",img_size=256, )

pretrain_generator(net_G, DataLoader(data, shuffle=True, batch_size=16), opt, criterion, 20)

torch.save(net_G.state_dict(), "pretrained_gen_256.pt")