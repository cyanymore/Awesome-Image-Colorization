
from tkinter.tix import Tree
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision as tv
import random 
from tqdm import tqdm
from torchvision import transforms
from src.utils import denorm, label2onehot, rgb2lab, set_deterministic

import kornia as K

grayscale = tv.transforms.Grayscale(num_output_channels=1)

def segmented_classes_extract(split, i, img_size, box, infrared=False):
    classes_temp, labels_temp = [],[]
    if infrared:
        path_or = split+"imgr"
        path_arrival = split + "maskscolr"
    else:
        path_or = split+"img"
        path_arrival = split + "maskscol"
    for idx in range(6):
        a =  i.replace(path_or,path_arrival)
        try:
            img_segm = Image.open(a.replace('.jpg', '_'+str(idx+1)+'.jpg')).convert('L')
        except:
            a = i.replace(split+"imgr",split + "maskscol") if infrared else i.replace(split+"img",split + "maskscolr")
            try:
                img_segm = Image.open(a.replace('.jpg', '_'+str(idx+1)+'.jpg')).convert('L')
                #WOW
            except:
                continue
        img_segm = img_segm.crop(box)

        # convert image to numpy array
        img_segm = np.asarray(img_segm)
        img_segm = cv2.resize(img_segm, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        test = label_preprocess(img_segm)
        c = np.array(idx+1) #class
        classes_temp.append(test)
        labels_temp.append(c)
    return classes_temp, labels_temp
def label_preprocess(data):
    # # data = data.astype(dtype=int)
    # # new_seg = np.zeros(data.shape, data.dtype)
    # # new_seg[(data > 55) & (data <= 70)] = 1
    # out = data.copy()

    # return out.astype(dtype=int)

    data = data.astype(dtype=int)
    # print(data.max())
    new_seg = np.zeros(data.shape, data.dtype)
    new_seg[data >15] = 1
    # plt.imshow(new_seg,cmap='gray')
    # plt.savefig('label_prep')
    # raise Exception
    return new_seg


# return image and optioanlly the masked image, in this case we want just to remove the white border
# so the idea is to put in black all the white pixels
def raw_preprocess(data, get_s=False, path_pair=None, img_size=256, ):
    box = (100, 100, 740, 612)
    if get_s:
        img_pair = Image.open(path_pair)
        img_pair = img_pair.crop(box)
        # convert image to numpy array
        img_pair = np.asarray(img_pair)
        #lasciamo qui sotto?????
        img_pair = cv2.resize(img_pair, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        data = data.astype(dtype=float)
        img_pair = img_pair.astype(dtype=float)
        out = data.copy()
        out_pair = img_pair.copy()
        
        #normalizzazione
        out = (out - out.min()) / (out.max() - out.min())
        out_pair = (out_pair - out_pair.min()) / (out_pair.max() - out_pair.min())

        return out, out_pair
    data = data.astype(dtype=float)
    # data[data<50] = 0
    out = data.copy()
    #normalizzazione?
    out = (out - out.min()) / (out.max() - out.min())

    return out

class DroneVeichleDataset(Dataset):
    def __init__(self, path="dataset", 
                       split='train', 
                       modals=('img','imgr'),
                       transforms=None,
                       img_size=128,
                       to_be_loaded=False,
                       colored_data=True,
                       paired_image=False,
                       lab = False,
                       classes=False,
                       debug = False,
                       single_mod=(False,'mod'),
                       remove_dark=False,
                       ):
        super(DroneVeichleDataset, self).__init__()
        
        if not to_be_loaded:
            cooo = 0
            self.remove_dark_samples = remove_dark

            self.single_mod = single_mod
            self.paired_image = paired_image
            self.lab = lab
            self.classes =classes
            box = (100, 100, 740, 612)
            self.img_size = img_size
            fold = split + "/"
            path1 = os.path.join(path, fold+ split+modals[0])
            path2 = os.path.join(path, fold + split+modals[1])
            list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted([os.path.join(path2, x) for x in os.listdir(path2)])
            raw_path = [] #contains RGB image real
            # print(len(list_path), list_path[200])
            if debug:
                list_path = list_path[:40]
            for x in list_path:
                if split+"imgr" in x:
                    c = np.array(0) #infrared
                    # tmp =  x.replace(split+"imgr",split+"masksr")
                elif split+"img" in x:
                    c = np.array(1)
                    # tmp =  x.replace(split+"img",split+"masks")
                else:
                    raise Exception('wrong path probably')
                raw_path.append([x,c])
                
            #########
            self.raw_dataset, self.raw_classes = [], []
            self.seg_mask_dataset, self.seg_classes_labels = [],[]
            #######
            self.transfroms = transforms

            for i,c in tqdm(raw_path): 
                if (c == 0 and not self.single_mod[0]) or (c == 0 and self.single_mod[0] and self.single_mod[1]=='ir'): #infrared
                    img = Image.open(i)

                    img = img.crop(box)
                    # convert image to numpy array
                    img = np.asarray(img)
                    if self.remove_dark_samples:
                        text = 1 if is_bright(img) else 0
                        if text!=1:
                            continue
                        cooo+=text
                    #lasciamo qui sotto?????
                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                    img, img_pair = raw_preprocess(img, get_s=True, path_pair=i.replace(split+"imgr",split+"img"), img_size = self.img_size)

                    if classes:
                        classes_seg, classes_labels = segmented_classes_extract(split, i, self.img_size, box, infrared=True)
                        if classes_labels ==[]:
                            print(i)
                            continue
                        self.raw_classes.append(classes_seg)
                        self.seg_classes_labels.append(classes_labels)
                    self.raw_dataset.append([(img, img_pair), c])

                    #?????
                    a =  i.replace(split+"imgr",split+"masksr")
                    img_segm = Image.open(a)
                    img_segm = img_segm.crop(box)

                    # convert image to numpy array
                    img_segm = np.asarray(img_segm)
                    img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                    self.seg_mask_dataset.append(label_preprocess(img_segm))
                elif (c==1 and not self.single_mod[0]) or (c==1 and self.single_mod[0] and self.single_mod[1]=='rgb'):
                    img = Image.open(i)
                    img = img.crop(box)

                    # convert image to numpy array
                    img = np.asarray(img)
                    if self.remove_dark_samples:
                        text = 1 if is_bright(img) else 0
                        if text!=1:
                            continue
                        cooo+=text

                    # img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                    #lasciamo qui sotto?????
                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                    img, img_pair = raw_preprocess(img, get_s=True, path_pair=i.replace(split+"img",split+"imgr"), img_size = self.img_size)

                    if classes:
                        classes_seg, classes_labels = segmented_classes_extract(split, i, self.img_size, box)
                        if classes_labels == []:
                            print(i)
                            continue
                        #dataset/val/valimg/01320.jpg
                        #dataset/val/valimg/01322.jpg
                        self.raw_classes.append(classes_seg)
                        self.seg_classes_labels.append(classes_labels)

                    self.raw_dataset.append([(img, img_pair), c])

                    #?????
                    a =  i.replace(split+"img",split+"masks")
                    img_segm = Image.open(a)
                    img_segm = img_segm.crop(box)

                    # convert image to numpy array
                    img_segm = np.asarray(img_segm)
                    img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                    self.seg_mask_dataset.append(label_preprocess(img_segm))


            self.split = split
            self.colored_data = colored_data
            assert len(self.raw_dataset) == len(self.seg_mask_dataset)
            if classes:
                assert len(self.raw_dataset) == len(self.raw_classes)

            print("DroneVeichle "+ split+ " data load success!")
            print("total size:{}".format(len(self.raw_dataset)))
            print(cooo)
    def __getitem__(self, item):
        img, paired_img, class_label, seg_mask = self.raw_dataset[item][0][0],\
                                                 self.raw_dataset[item][0][1], \
                                                 self.raw_dataset[item][1], \
                                                 self.seg_mask_dataset[item]
        if self.classes:
            seg_classes = self.raw_classes[item]
            seg_labels = self.seg_classes_labels[item]
        else:
            t_imgs_classes = []
            seg_labels = []
        if img.shape[0]!=self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            seg_mask = cv2.resize(seg_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            paired_img = cv2.resize(paired_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            if self.classes:
                seg_classes = [cv2.resize(class_seg, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST) for class_seg in seg_classes]
        #trhee channels for seg mask
        # print(seg_mask.shape, img.shape)
        if len(img.shape)>2:
            seg_mask_3 = np.repeat(seg_mask[...,None],3,axis=2)
            t_img = img * seg_mask_3
            if self.classes:
                t_imgs_classes = [img * np.repeat(class_seg[...,None],3,axis=2)  for class_seg in seg_classes]
        else:
            t_img = img * seg_mask
            if self.classes:
                t_imgs_classes = [img * class_seg  for class_seg in seg_classes]
            # print(';infrared', class_label)
        if self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                seg_mask = cv2.flip(seg_mask, 1)
                paired_img = cv2.flip(paired_img, 1)
                t_img = cv2.flip(t_img, 1)
                if self.classes:
                    seg_classes = [cv2.flip(class_seg, 1) for class_seg in seg_classes]
                    t_imgs_classes = [cv2.flip(t_imgs_class,1) for t_imgs_class in t_imgs_classes]
        # if self.lab:
        #     img = img/255
        #     t_img = t_img/255
        #     paired_img = paired_img/255
        #     if self.classes:
        #         t_imgs_classes = [t_imgs_class/255 for t_imgs_class in t_imgs_classes]
        if not self.lab:
            # scale to [-1,1]
            img = (img - 0.5) / 0.5
            t_img = (t_img - 0.5) / 0.5
            paired_img = (paired_img - 0.5) / 0.5
            if self.classes:
                t_imgs_classes = [(t_imgs_class- 0.5) / 0.5 for t_imgs_class in t_imgs_classes]
        if len(img.shape)>2:
            img, t_img, paired_img, seg_mask, class_label, t_imgs_classes = self.get_item_rgb(img, t_img, paired_img, seg_mask, class_label,t_imgs_classes if t_imgs_classes is not [] else None)
            if self.lab:
                img = rgb2lab(img)
                t_img = rgb2lab(t_img)
                paired_img = rgb2lab(paired_img)
                if self.classes:
                    t_imgs_classes = [rgb2lab(t_imgs_class) for t_imgs_class in t_imgs_classes]
                
        else:
            img, t_img, paired_img, seg_mask, class_label, t_imgs_classes = self.get_item_grey(img, t_img, paired_img, seg_mask, class_label, t_imgs_classes if t_imgs_classes is not [] else None)
            if self.lab:
                img = rgb2lab(img)
                t_img = rgb2lab(t_img)
                paired_img = rgb2lab(paired_img)
                if self.classes:
                    t_imgs_classes = [rgb2lab(t_imgs_class) for t_imgs_class in t_imgs_classes]
        
        if self.classes:
            seg_labels = [torch.from_numpy(lab) for lab in seg_labels]
            if seg_labels == []:
                plt.imshow(denorm(img).cpu().numpy().transpose(1,2,0))
                plt.title('target image')
                plt.savefig('a') 
                print(item) 
                raise Exception
            while(len(seg_labels)) < 6:
                seg_labels.append(torch.tensor(0))
                t_imgs_classes.append(torch.zeros(t_img.shape))
            
            return img, t_img, paired_img, seg_mask, class_label, torch.stack(t_imgs_classes), torch.stack(seg_labels)

        return img, t_img, paired_img, seg_mask, class_label, #if training mode torch.stack(t_imgs_classes), torch.stack(seg_labels)
        
    def __len__(self):
        return len(self.raw_dataset)

    def load_dataset(self, path="dataset", split='train', idx = "0", img_size=128, colored_data = True, paired_image=False, lab=False, classes=False,
                        debug = False,
                       single_mod=(False,'mod'),
                       remove_dark=False):
        self.seg_mask_dataset = torch.load(f"{path}/{idx}_{split}_seg_mask_dataset.pt")
        self.raw_dataset = torch.load(f"{path}/{idx}_{split}_raw_dataset.pt")
        self.img_size = img_size
        self.split=split
        self.colored_data = colored_data
        self.paired_image = paired_image
        self.lab = lab
        self.classes = classes
        self.debug = debug
        self.single_mod = single_mod
        self.remove_dark = remove_dark

    def get_item_rgb(self, img, t_img, paired_img, seg_mask, class_label,t_imgs_classes):
        if not self.colored_data:
            img = grayscale(torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1))
            t_img = grayscale(torch.from_numpy(t_img).type(torch.FloatTensor).permute(2, 0, 1))
            paired_img = grayscale(torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1))
            if t_imgs_classes is not None:
                t_imgs_classes = [grayscale(torch.from_numpy(t_img_class).type(torch.FloatTensor).permute(2, 0, 1)) for t_img_class in t_imgs_classes]

        else:
            img, t_img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1), \
                torch.from_numpy(t_img).type(torch.FloatTensor).permute(2, 0, 1)
            paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1) if len(paired_img.shape) ==3 else \
                            torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            if t_imgs_classes is not None:
                t_imgs_classes = [torch.from_numpy(t_img_class).type(torch.FloatTensor).permute(2, 0, 1) for t_img_class in t_imgs_classes]

        return img, \
            t_img , \
            paired_img, \
            torch.from_numpy(seg_mask).type(torch.LongTensor).unsqueeze(dim=0), \
            torch.from_numpy(class_label).type(torch.FloatTensor), \
            t_imgs_classes

    def get_item_grey(self, img, t_img, paired_img, seg_mask, class_label, t_imgs_classes):
        if not self.colored_data:
            img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0)
            t_img = torch.from_numpy(t_img).type(torch.FloatTensor).unsqueeze(dim=0)
            paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0)
            if t_imgs_classes is not None:
                t_imgs_classes = [torch.from_numpy(t_img_class).type(torch.FloatTensor).unsqueeze(dim=0) for t_img_class in t_imgs_classes]

        else:
            img =  torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            t_img = torch.from_numpy(t_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            if self.paired_image:
                paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1) if len(paired_img.shape) ==3 else \
                            torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
                # paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1)
            else:
                paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1)
            if t_imgs_classes is not None:
                t_imgs_classes = [torch.from_numpy(t_img_class).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1) for t_img_class in t_imgs_classes]

        return img, \
            t_img, \
            paired_img, \
            torch.from_numpy(seg_mask).type(torch.LongTensor).unsqueeze(dim=0), \
            torch.from_numpy(class_label).type(torch.FloatTensor), \
            t_imgs_classes



# testing_dataset()


class DroneVeichleDatasetPreTraining(Dataset):
    def __init__(self,path="dataset", split='train', modals=('img','imgr'),transforms=None, img_size=128, to_be_loaded=False, colored_data=False):
        super(DroneVeichleDatasetPreTraining, self).__init__()
        
        box = (100, 100, 740, 612)
        self.img_size = img_size
        fold = split + "/"
        path1 = os.path.join(path, fold+ split+modals[0])
        path2 = os.path.join(path, fold + split+modals[1])
        list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted([os.path.join(path2, x) for x in os.listdir(path2)])
        raw_path = [] #contains RGB image real
        # print(len(list_path), list_path[200])
        random.shuffle(list_path)
        for x in list_path[:20000]:

            if split+"imgr" in x:
                c = np.array(0) #infrared
                # tmp =  x.replace(split+"imgr",split+"masksr")
            elif split+"img" in x:
                c = np.array(1)
                # tmp =  x.replace(split+"img",split+"masks")
            else:
                raise Exception('wrong path probably')
            raw_path.append([x,c])
            
        #########
        self.dataset = []
        #######
        self.transfroms = transforms

        for i,c in tqdm(raw_path): 
            img = Image.open(i)
            img = img.crop(box)

            # convert image to numpy array
            img = np.asarray(img)
            # img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            #lasciamo qui sotto?????
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img = raw_preprocess(img )

            #?????
            if c==0:
                a = i.replace(split+"imgr",split+"img")
                irr = True
            else:
                a = i.replace(split+"img",split+"imgr")
                irr = False

            img_2 = Image.open(a)
            img_2 = img_2.crop(box)

            # convert image to numpy array
            img_2 = np.asarray(img_2)
            img_2 = cv2.resize(img_2, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_2 = raw_preprocess(img_2)
            if irr:
                self.dataset.append((img, img_2, np.array(1)))
            else:
                self.dataset.append((img_2, img, np.array(1)))

            irr = False
        self.split = split
        self.colored_data = colored_data
        print("DroneVeichle "+ split+ " data load success!")
        print("total size:{}".format(len(self.dataset)))
            
    def __getitem__(self, item):
        img, img_2 = self.dataset[item][0], self.dataset[item][1]
        if img.shape[0]!=self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_2 = cv2.resize(img_2, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        #trhee channels for seg mask
        # print(seg_mask.shape, img.shape)
        
        if self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                img_2 = cv2.flip(img_2, 1)

        #  scale to [-1,1]
        img = (img - 0.5) / 0.5
        img_2 = (img_2 - 0.5) / 0.5

        lab = torch.from_numpy(self.dataset[item][2]).type(torch.FloatTensor)

        if len(img.shape)>2:
            img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1)
        else:
            img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
        if len(img_2.shape)>2:
            img_2 = torch.from_numpy(img_2).type(torch.FloatTensor).permute(2, 0, 1)
        else:    
            img_2 = torch.from_numpy(img_2).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
        
        return img, img_2, lab

    def __len__(self):
        return len(self.dataset)





def save_tensors_dataset(path="dataset", split="train", slices=19, max_length_slices=2000, img_size=256, my_folder = "dataset/tensors/tensors_paired", classes=False):
    modals=('img','imgr')
    fold = split + "/"
    path1 = os.path.join(path, fold+ split+modals[0])
    path2 = os.path.join(path, fold + split+modals[1])
    list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted([os.path.join(path2, x) for x in os.listdir(path2)])
    print(len(list_path))
    for idx in tqdm(range(slices)):
        random.shuffle(list_path)
        l_o = list_path[:max_length_slices]
        dt = DroneVeichleDataset(l_o,split=split, img_size=img_size, classes=classes, remove_dark=True)
        idx = str(idx)

        torch.save(dt.raw_dataset, f"{my_folder}/{idx}_{split}_raw_dataset.pt")
        torch.save(dt.seg_mask_dataset, f"{my_folder}/{idx}_{split}_seg_mask_dataset.pt")
        if classes:
            torch.save(dt.raw_classes, f"{my_folder}/{idx}_{split}_raw_classes_dataset.pt")

        list_path = list(set(list_path)-set(l_o))
        print(len(list_path), " remaining samples")


# save_tensors_dataset(path="dataset", split="train", slices=19, max_length_slices=2000, img_size=256,  my_folder = "dataset/tensors/tensors_classes")
# save_tensors_dataset(path="dataset", split="val", slices=2, max_length_slices=2000, img_size=256,  my_folder = "dataset/tensors/tensors_classes")

class DefaultDataset(Dataset):
    def __init__(self, root, img_size=256, transform=None, kaist=False, remove_dark=False):
        if not kaist:
            self.samples = os.listdir(root)
            self.samples.sort()
            if "ground_truth" in self.samples:
                self.samples.remove("ground_truth")
            self.root = root
        else:
            imageset_txt = "dataset/kaist-cvpr15/imageSets/test-all-20.txt" if "val" in root else "dataset/kaist-cvpr15/imageSets/train-all-04.txt"
            with open(imageset_txt) as f:
                llp = [line.strip() for line in f.readlines()]
            path1,path2 = [],[]
            for sett in llp:
                splitted = sett.split("/")
                splitted.insert(-1,"lwir")
                final = "/".join(splitted) +".jpg"
                path1.append(final)
                path2.append(final.replace("lwir","visible"))
            self.samples = (path1+path2)
            if "ground_truth" in self.samples:
                self.samples.remove("ground_truth")
            self.root = "dataset/kaist-cvpr15/images/"
        self.transform = transform
        self.targets = None
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.box = (100, 100, 740, 612)

        self.transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])
        if remove_dark:
            with open('dataset/val/dark_samples.txt','r') as f:
                my_list=f.readlines()
            temp = [line[:-1] for line in my_list]
            self.samples= list(set(self.samples) -set(temp))
    
    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(os.path.join(self.root,fname)).convert('RGB')
        img = img.crop(self.box)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)



def is_bright(image, dim=10, thresh=0.5):
    # Resize image to 10x10
    if len(image.shape) <3:
        image = np.repeat(image[...,None],3,axis=2)
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh

class KAISTDataset(Dataset):
    def __init__(self, path="dataset", split='train', modals=('img','imgr'),transforms=None, img_size=128, to_be_loaded=False, colored_data=True, paired_image=False):
        super(KAISTDataset, self).__init__()
        imageset_txt = "dataset/kaist-cvpr15/imageSets/test-all-20.txt" if split == "val" else  "dataset/kaist-cvpr15/imageSets/train-all-04.txt"
        if not to_be_loaded:
            self.paired_image = paired_image
            self.img_size = img_size
            root = "dataset/kaist-cvpr15/images/"
            # setss = os.listdir(root)
            # for sett in setss:
            #     vvs = os.listdir(os.path.join(root,sett))
            #     for vv in vvs:
            with open(imageset_txt) as f:
                llp = [line.strip() for line in f.readlines()]
            print(len(llp))
            path1,path2, raw_path = [],[],[]
            for sett in llp:
                splitted = sett.split("/")
                splitted.insert(-1,"lwir")
                final = "/".join(splitted) +".jpg"
                path1.append(final)
                path2.append(final.replace("lwir","visible"))
            list_path = sorted(path1+path2)
            random.shuffle(list_path)
            for x in list_path[:1000]:
                if "lwir" in x:
                    c = np.array(0) #infrared
                    # tmp =  x.replace(split+"imgr",split+"masksr")
                elif "visible" in x:
                    c = np.array(1)
                    # tmp =  x.replace(split+"img",split+"masks")
                else:
                    raise Exception('wrong path probably')
                raw_path.append([x,c])
                            
            #########
            self.raw_dataset = []
            self.seg_mask_dataset = []
            #######
            self.transfroms = transforms

            for i,c in tqdm(raw_path): 
                if c == 0: #infrared
                    img = Image.open(root+i)
                    # convert image to numpy array
                    img = np.asarray(img)
                    #lasciamo qui sotto?????
                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                    img, img_pair = self.raw_preprocess(img, get_s=True, path_pair=root+ i.replace("lwir","visible"), img_size = self.img_size)

                    self.raw_dataset.append([(img, img_pair), c])
                    #?????
                    # a =  i.replace(split+"imgr",split+"masksr")
                    # img_segm = Image.open(root+a)

                    # # convert image to numpy array
                    # img_segm = np.asarray(img_segm)
                    # img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                    # self.seg_mask_dataset.append(self.label_preprocess(img_segm))
                elif c==1:
                        img = Image.open(root+i)

                        # convert image to numpy array
                        img = np.asarray(img)
                        # img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                        #lasciamo qui sotto?????
                        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                        img, img_pair = self.raw_preprocess(img, get_s=True, path_pair=root+i.replace("visible","lwir"), img_size = self.img_size)

                        self.raw_dataset.append([(img, img_pair), c])
                        #?????
                        # a =  i.replace(split+"img",split+"masks")
                        # img_segm = Image.open(root+a)

                        # # convert image to numpy array
                        # img_segm = np.asarray(img_segm)
                        # img_segm = cv2.resize(img_segm, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                        # self.seg_mask_dataset.append(self.label_preprocess(img_segm))


            self.split = split
            self.colored_data = colored_data
            print("KAIST "+ split+ " data load success!")
            print("total size:{}".format(len(self.raw_dataset)))
            
    def __getitem__(self, item):
        img, paired_img, class_label = self.raw_dataset[item][0][0],\
                                                 self.raw_dataset[item][0][1], \
                                                 self.raw_dataset[item][1], \

        if img.shape[0]!=self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            paired_img = cv2.resize(paired_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        
        if self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                paired_img = cv2.flip(paired_img, 1)
        #  scale to [-1,1]
        img = (img - 0.5) / 0.5

        if len(img.shape)>2:
            return self.get_item_rgb(img, paired_img, class_label)
        else:
            return self.get_item_grey(img, paired_img, class_label)

        
    def __len__(self):
        return len(self.raw_dataset)

    def load_dataset(self, path="dataset", split='train', idx = "0", img_size=128, colored_data = True, paired_image=False, lab=False):
        self.seg_mask_dataset = torch.load(f"{path}/{idx}_{split}_label_dataset.pt")
        self.raw_dataset = torch.load(f"{path}/{idx}_{split}_raw_dataset.pt")
        self.img_size = img_size
        self.split=split
        self.colored_data = colored_data
        self.paired_image = paired_image
        self.lab = lab
    def get_item_rgb(self, img, paired_img, class_label):
        if not self.colored_data:
            img = grayscale(torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1))
            paired_img = grayscale(torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1))
        else:
            img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1)
            paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1) if len(paired_img.shape) ==3 else \
                            torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            img=rgb2lab(img)
            paired_img = rgb2lab(paired_img)
        return img, \
            paired_img, \
            torch.from_numpy(class_label).type(torch.FloatTensor)

    def get_item_grey(self, img, paired_img, class_label):
        if not self.colored_data:
            img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0)
            paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0)
        else:
            img =  torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
            if self.paired_image:
                paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1) if len(paired_img.shape) ==3 else \
                            torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)
                # paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).permute(2, 0, 1)
            else:
                paired_img = torch.from_numpy(paired_img).type(torch.FloatTensor).unsqueeze(dim=0).repeat(3,1,1)

        return img, \
            paired_img, \
            torch.from_numpy(class_label).type(torch.FloatTensor)

    def label_preprocess(self,data):
        # # data = data.astype(dtype=int)
        # # new_seg = np.zeros(data.shape, data.dtype)
        # # new_seg[(data > 55) & (data <= 70)] = 1
        # out = data.copy()

        # return out.astype(dtype=int)

        data = data.astype(dtype=int)
        # print(data.max())
        new_seg = np.zeros(data.shape, data.dtype)
        new_seg[data >15] = 1
        # plt.imshow(new_seg,cmap='gray')
        # plt.savefig('label_prep')
        # raise Exception
        return new_seg


    # return image and optioanlly the masked image, in this case we want just to remove the white border
    # so the idea is to put in black all the white pixels
    def raw_preprocess(self, data, get_s=False, path_pair=None, img_size=256, ):
        if get_s:
            img_pair = Image.open(path_pair)
            # convert image to numpy array
            img_pair = np.asarray(img_pair)
            #lasciamo qui sotto?????
            img_pair = cv2.resize(img_pair, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            
            data = data.astype(dtype=float)
            img_pair = img_pair.astype(dtype=float)
            out = data.copy()
            out_pair = img_pair.copy()
            
            #normalizzazione
            out = (out - out.min()) / (out.max() - out.min())
            out_pair = (out_pair - out_pair.min()) / (out_pair.max() - out_pair.min())

            return out, out_pair
        data = data.astype(dtype=float)
        # data[data<50] = 0
        out = data.copy()
        #normalizzazione?
        out = (out - out.min()) / (out.max() - out.min())

        return out
def testing_dataset():
    dt = DroneVeichleDataset(split="val", img_size=128, classes=True)
    # dt = ChaosDataset_Syn_new(path="../QWT/TarGAN/datasets/chaos2019")
    syn_loader = DataLoader(dt, shuffle=True)
    print(len(syn_loader))
    for epoch, (x_real, t_img, paired_img, mask, label_org, classes_seg, lab_seg) in enumerate(syn_loader):
    # for epoch, (x_real, paired_img, label_org) in enumerate(syn_loader):

        plt.axis('off')
        plt.subplot(241)
        plt.imshow(denorm(x_real).squeeze().cpu().numpy().transpose(1,2,0))
        plt.title('real image')
        plt.subplot(242)
        plt.imshow(denorm(t_img).squeeze().cpu().numpy().transpose(1,2,0))
        plt.title('target image')
        plt.subplot(243)
        plt.imshow(paired_img.squeeze().cpu().numpy().transpose(1,2,0),)
        plt.title('paired image')
        plt.subplot(244)
        plt.imshow(denorm(mask).squeeze().cpu().numpy(),cmap=plt.get_cmap('gray'))
        plt.title('target mask')
        plt.savefig('test'+str(epoch))
        plt.show()

        #print(lab_seg)
        # lab_seg = label2onehot(lab_seg, 6)

        # for classes, l_seg in zip(classes_seg,lab_seg):
        #     # print(classes.shape)
        #     idxs = [(l_seg[i] ==1) if i>0 else False for i in range(l_seg.size(0))]
        #     idx = idxs.index(True) if len(list((filter(lambda score: score == True, idxs)))) == 1 else False
        #     if idx == False:
        #         check = False
        #         while check ==False:
        #             idx = random.randint(1, len(idxs)-1)
        #             check = idxs[idx]
            
        #     plt.imshow(denorm(classes)[idx].cpu().numpy().transpose(1,2,0))
        #     plt.title('target image'+str(idx))
        #     #plt.savefig('a'+str(epoch))
        # #assume batch size = 1, show every target even if not exists
        # for idx in range(classes_seg.size(1)):
        #     plt.imshow(denorm(classes_seg)[0][idx].cpu().numpy().transpose(1,2,0))
        #     plt.title('target image'+str(l_seg[0]))
        #     #plt.savefig('b'+str(idx))
if __name__ == "__main__":
    print()
    #testing_dataset()
    set_deterministic(888)
    # save_tensors_dataset(path="dataset", split="train", slices=19, max_length_slices=2000, img_size=256,  my_folder = "dataset/tensors/tensors_classes")
    # save_tensors_dataset(path="dataset", split="val", slices=2, max_length_slices=2000, img_size=256,  my_folder = "dataset/tensors/tensors_classes")
