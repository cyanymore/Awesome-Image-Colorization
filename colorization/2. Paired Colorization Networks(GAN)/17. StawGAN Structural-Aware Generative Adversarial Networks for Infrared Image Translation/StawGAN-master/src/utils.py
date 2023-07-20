import os
from smtpd import DebuggingServer
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from pathlib import Path
import wandb
import torchvision.utils as vutils
import json
import kornia as K

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename,args):
    x = denormalize(x)

    # IF BATCH
    if len(x.shape) == 4:
        if x.size(1) == 3: # non quat
            x = x
        else: #quat
            #x = x[:,1:4,:,:]
            x = x[:,0:3,:,:]
    
        sample_dir = " ".join(filename.replace(args.experiment_name,"").replace(".jpg","").split("_")[1:])
        if args.mode=="train":
            iters = str(int(filename.replace(args.experiment_name,"").split("/")[3].split("_")[0]))

    # IS ONLY ONE PHOTO
    else:
        if x.size(0) == 3: # non quat
            x = x
        else: #quat
            #x = x[:,1:4,:,:]
            x = x[0:3,:,:]
        #iters = str(int(filename.split("/")[9].split("_")[0]))
    
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
    if len(x.shape) == 4 and args.mode=="train":
        wandb.log({sample_dir: wandb.Image(filename,caption=iters)},commit=False)

def loss_filter(mask,device="cuda" if torch.cuda.is_available() else "cpu"):
    list = []
    for i, m in enumerate(mask):
        if torch.any(m == 1):
            list.append(i)
    index = torch.tensor(list, dtype=torch.long).to(device)
    return index

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def denorm(x):
    res = (x + 1.) / 2.
    res.clamp_(0, 1)
    return res

def renorm(x):
    res = (x - 0.5) / 0.5
    res.clamp_(-1, 1)
    return res


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    for i in range(batch_size):
        out[i, labels[i].long()] = 1
    return out

def getLabel(imgs, device, index, c_dim=2):
    syn_labels = torch.zeros((imgs.size(0), c_dim)).to(device)
    syn_labels[:, index] = 1.
    return syn_labels

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def save_state_net(net, args, index, optim=None, experiment_name="test"):
    save_path = os.path.join(args.save_path, args.experiment_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, args.net_name)
    torch.save(net.state_dict(), save_file + '_' + str(index) +"_"+experiment_name + '.pkl')
    if optim is not None:
        torch.save(optim.state_dict(), save_file + '_optim_' + str(index)+ "_" + experiment_name + '.pkl')
    if not os.path.isfile(save_path + '/outputs.txt'):
        with open(save_path + '/outputs.txt', mode='w') as f:
            argsDict = args.__dict__;
            f.writelines(args.experiment_name + '\n')
            for i in argsDict.keys():
                f.writelines(str(i) + ' : ' + str(argsDict[i]) + '\n')

def load_state_net(args, net, net_name, index, optim=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    save_path = os.path.join(args.save_path, args.experiment_name)
    if not os.path.isdir(save_path):
        raise Exception("wrong path")
    save_file = os.path.join(save_path, net_name)
    if net is not None:
        net.load_state_dict(torch.load(save_file + '_' + str(index)+ "_"  + args.experiment_name + '.pkl', map_location=device))
    if optim is not None:
        optim.load_state_dict(torch.load(save_file + '_optim_' + str(index)+ "_"  + args.experiment_name + '.pkl'))
    # print(net)
    return net, optim

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)



def plot_images(nets, syneval_dataset, device, c_dim, wavelet_type, lab, classes, debug, single_mod, preloaded_data):
    # fig = plt.figure(dpi=120)
    idx = random.randint(0,20) if debug or single_mod or preloaded_data else random.randint(1460,1485)
    with torch.no_grad():
        img = syneval_dataset[idx][0]
        pair_img = syneval_dataset[idx][2]
        if classes[0]:
            t_imgs_classes_org = syneval_dataset[idx][5]
            classes_org = syneval_dataset[idx][6].unsqueeze(0)
            rand_idx_classes = torch.randperm(classes_org.size(0))
            classes_trg = classes_org[rand_idx_classes]
            c_classes_trg = (label2onehot(classes_trg, 6) - torch.tensor([1,0,0,0,0,0])).to(device)
            #classes_org torch.tensor([1,0,0,0,0,0]),([1,1,0,0,0,0])]
            #  ho una macchina al t_imgs_classes[i][0], ho una macchina in t_imgs_classes[i][0] e un truck in t_imgs_classes[i][1] 
            t_imgs_classes_org = t_imgs_classes_org.unsqueeze(0).to(device)
            t_imgs_classes_trg = t_imgs_classes_org[rand_idx_classes].to(device)
            if classes[1]:
                yy_trg, style_trg, x_segm_valid = get_style(nets,y_trg=c_classes_trg, x_segm= t_imgs_classes_trg, x_real=img.unsqueeze(0).to(device))
        img = img.unsqueeze(dim=0).to(device)
        try:
            #trg_segm = syneval_dataset[idx][3]
            trg_orig = syneval_dataset[idx][1]
            trg_orig = trg_orig.unsqueeze(dim=0).to(device)
        except:
            pred_t1_img = nets.netG_use(img, c=getLabel(img, device, 0, c_dim), mode="kaist", wav_type=wavelet_type)
            pred_t2_img = nets.netG_use(img, c=getLabel(img, device, 1, c_dim), mode="kaist", wav_type=wavelet_type)
            return denorm(img).cpu(), \
            denorm(pred_t1_img).cpu(), \
            denorm(pred_t2_img).cpu(), \
        # print(getLabel(img, device, 0, args.c_dim).shape,img.shape)
        pred_t1_img, pred_t1_targ = nets.netG_use(img, trg_orig, c=getLabel(img, device, 0, c_dim), 
                                                wav_type=wavelet_type, style=None if not classes[1] else style_trg, class_label=c_classes_trg if classes[0] and not classes[1] else None)
        pred_t2_img, pred_t2_targ = nets.netG_use(img, trg_orig, c=getLabel(img, device, 1, c_dim), 
                                                wav_type=wavelet_type, style=None if not classes[1] else style_trg, class_label=c_classes_trg if classes[0] and not classes[1] else None)

    if lab:
        return (lab2rgb(img.cpu()[:,:1],img.cpu()[:,1:] )), \
        (lab2rgb(pred_t1_img.cpu()[:,:1],pred_t1_img.cpu()[:,1:])), \
        (lab2rgb(pred_t2_img.cpu()[:,:1],pred_t2_img.cpu()[:,1:])), \
        lab2rgb(trg_orig.cpu()[:,:1],trg_orig.cpu()[:,1:]), \
        lab2rgb(pred_t1_targ.cpu()[:,:1],pred_t1_targ.cpu()[:,1:]), \
        lab2rgb(pred_t2_targ.cpu()[:,:1],pred_t2_targ.cpu()[:,1:]), \
        (lab2rgb(pair_img.unsqueeze(0).cpu()[:,:1],pair_img.unsqueeze(0).cpu()[:,1:])), \
        lab2rgb(x_segm_valid.cpu()[:,:1],x_segm_valid.cpu()[:,1:]) if classes[1] else None

    return denorm(img).cpu(), \
            denorm(pred_t1_img).cpu(), \
            denorm(pred_t2_img).cpu(), \
            denorm(trg_orig).cpu(), \
            pred_t1_targ.cpu(), \
            pred_t2_targ.cpu(), \
            denorm(pair_img).cpu(), \
            denorm(x_segm_valid).cpu() if classes[1] else None


def load_nets(args,nets,sepoch, optims):
    optims_list= list(optims.keys())
    for i,net in enumerate(nets.keys()):
        print("loading", net)
        net_check = net if "use" in net else net.replace("_", "")
        load_state_net(args,nets[net], net_check, sepoch, optim=optims[optims_list[i]])

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def get_valid_targets(y_trg, x_segm):
    x_segm_valid,y_trg_valid = [],[]
    for b in range(y_trg.size(0)): #batch size
        glob_idx = 0
        found = False
        x_segm_valid_batch, y_trg_valid_batch = [],[]
        for idx in range(1,y_trg.size(1)): #scorri tensori fatti cosi torch.tensor([1., 0., 0., 0., 0.]) , skippi il primo che tanto é targ nullo, 
            if torch.equal(y_trg[b][idx], torch.tensor(0.).to(device)):
                continue
            else:
                found=True
                x_segm_valid_batch.append(x_segm[b,glob_idx])
                y_trg_valid_batch.append(torch.tensor(idx))
                glob_idx+=1
        if found:
            rand_id = random.randint(0,len(x_segm_valid_batch)-1)
            x_segm_valid.append(x_segm_valid_batch[rand_id])
            y_trg_valid.append(y_trg_valid_batch[rand_id])
            continue
        else:
            # print(x_segm.shape, x_segm_valid.shape)
            print(y_trg[b])
            raise Exception("zeros")
        #yy_trg = onehot2label(yy_trg.unsqueeze(0)).squeeze(0).long()
    return x_segm_valid,y_trg_valid

def get_style(nets,y_trg, x_segm, x_real=None):
    x_segm_valid,y_trg_valid = get_valid_targets(y_trg, x_segm)
    # if "debug" in args.mode:
    #     debugging_photo(x_segm_valid)
    x_segm_valid = torch.stack(x_segm_valid).to(device)
    y_trg_valid = torch.stack(y_trg_valid).to(device)
    if x_real is not None:
        s_trg = nets.netSE(x_real, y_trg_valid) #changed
    else:
        s_trg = nets.netSE(x_segm_valid, y_trg_valid) #changed

    return y_trg_valid, s_trg, x_segm_valid


#list
def debugging_photo(x_segm_valid):
    for i,x_segm in enumerate(x_segm_valid):
        plt.axis('off')
        plt.subplot(2,4,i+1)
        plt.imshow(denorm(x_segm).squeeze().cpu().numpy().transpose(1,2,0))
        plt.title('real image')
    
    plt.savefig('test'+str(1))


from skimage import color  # used for lab2rgb

def lab2rgb(L, AB):
    """Convert an Lab tensor image to a RGB numpy output
    Parameters:
        L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
        AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor arrqay)
    Returns:
        rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
    """
    # print(L.shape)
    # if len(L.shape)>2:
    #     L.squeeze()
    #     AB.squeeze()
    # print(L.shape)
    AB2 = AB * 110.0
    L2 = (L + 1.0) * 50.0
    Lab = torch.cat([L2, AB2], dim=1)
    Lab = Lab[0].data.cpu().float().numpy()
    Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
    rgb = color.lab2rgb(Lab) * 255
    return rgb

from PIL import Image
from torchvision import transforms
def rgb2lab(img_tensor=None, path=None):
    """Return a data point and its metadata information.
    Parameters:

    Returns a dictionary that contains A, B, A_paths and B_paths
        A (tensor) - - the L channel of an image
        B (tensor) - - the ab channels of the same image
        A_paths (str) - - image paths
        B_paths (str) - - image paths (same as A_paths)
    """
    if path is not None:
        im = Image.open(path).convert('RGB')
        im = np.array(im)
    else:
        im = img_tensor.numpy().transpose(1,2,0)
    lab = color.rgb2lab(im).astype(np.float32)
    lab_t = transforms.ToTensor()(lab)
    L = lab_t[[0], ...] / 50.0 - 1.0
    ab = lab_t[[1, 2], ...] / 110.0
    return torch.cat([L,ab], dim=0)





def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False