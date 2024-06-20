import numpy as np
import json
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
#                (car red)   (feright_car verdechiaro)    (truck viola)      (bus verdescuro)     (van giallo)
color_palette = [(255,0,0), (128,0,128),                    (0,192,0), (0,100,0), (200,200,0)]


def poly_mask(annotations, bbox,*vertices, value=0):
    """
    Create a mask array filled with 1s inside the polygon and 0s outside.
    The polygon is a list of vertices defined as a sequence of (column, line) number, where the start values (0, 0) are in the
    upper left corner. Multiple polygon lists can be passed in input to have multiple,eventually not connected, ROIs.
        column, line   # x, y
        vertices = [(x0, y0), (x1, y1), ..., (xn, yn), (x0, y0)] or [x0, y0, x1, y1, ..., xn, yn, x0, y0]
    Note: the polygon can be open, that is it doesn't have to have x0,y0 as last element.

    adapted from: https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/64876117#64876117
    :param shape:    (tuple) shape of the output array (height, width)
    :param vertices: (list of tuples of int): sequence of vertices defined as
                                            [(x0, y0), (x1, y1), ..., (xn, yn), (x0, y0)] or
                                            [x0, y0, x1, y1, ..., xn, yn, x0, y0]
                                            Multiple lists (for multiple polygons) can be passed in input
    :param value:    (float or NAN)      The masking value to use (e.g. a very small number). Default: np.nan
    :return:         (ndarray) the mask array
    """
    width, height = 840,712 #shape[::-1]
    
    #width, height = 640,512
    # create a binary image
    img = Image.new(mode='RGB', size=(width, height), color=0)  # mode L = 8-bit pixels, black and white
    draw = ImageDraw.Draw(img)
    # draw polygons
    for i,polygon in enumerate(vertices):
        draw.polygon(polygon, outline=1, fill=color_palette[annotations["seg"][i]-1])
    
    for i,box in enumerate(bbox):
        draw.rectangle(box, outline=1, fill=color_palette[annotations["bbox"][i]-1])
        # replace 0 with 'value'
    mask = np.array(img).astype('uint8')
    mask[np.where(mask == 0)] = value
    
    return mask



#masks = poly_mask(None, bbox, *polygons)


def create_masks(split="train", ir=False):
    ir_string = "r" if ir else ""
    with open("dataset/"+split+"/output_"+split+"_img"+ir_string+".json") as js:
        diz = json.load(js)

    #image = {'file_name': '11882.jpg', 'height': 712, 'width': 840, 'id': 11882}
    #annotation = {'iscrowd': 0, 'segmentation': [[692, 233, 729, 228, 732, 243, 695, 247]], 'category_id': 1, 'ignore': 0, 'bbox': [], 'image_id': 11882, 'id': 286792}, 
    dct = {}
    for image in tqdm(diz['images'], total=len(diz['images'])):
        anns,bboxs = [], []
        cats = {}
        cats["seg"],cats["bbox"] = [],[]
        
        for ann in diz['annotations']:
            if ann['image_id'] == image['id']:
                if ann['segmentation'] != []:
                    # print(ann)
                    anns.append(ann['segmentation'][0])
                    cats["seg"].append(ann['category_id'])
                elif ann['bbox'] != []:
                    # print(ann)
                    bboxs.append(ann['bbox'])
                    cats["bbox"].append(ann['category_id'])
        # print("----end---")
        for ids in range(6):
            anns_temp = [anns[i] for i,seg in enumerate(cats["seg"]) if seg == ids]
            bboxs_temp = [bboxs[i] for i,bbox in enumerate(cats["bbox"]) if bbox == ids]
            if anns_temp != [] or bboxs_temp != []:
                masks = poly_mask({"seg": [ids for i in range(len(anns_temp))], "bbox":[ids for i in range(len(bboxs_temp))]}, bboxs_temp, *anns_temp)
                imageio.imwrite('dataset/'+split+'/'+split+'maskscol'+ir_string+'/'+image['file_name'].replace(".jpg","")+"_"+str(ids)+".jpg", masks)

        masks = poly_mask(cats, bboxs, *anns)
        imageio.imwrite('dataset/'+split+'/'+split+'maskscol'+ir_string+'/'+image['file_name'], masks)

        # print(masks.shape)
        # plt.axis('off')
        # plt.gray()
        # plt.imshow(masks)
        # if len(bboxs)+len(anns) >20:
            # print(len(bboxs)+len(anns))
            # plt.show()
        #plt.savefig('/home/jary/Documents/luigi/captainwho/droneveichle/dataset/train/trainmasks/'+image['file_name'], dpi=120)
        # dct[image['file_name'].replace(".jpg","")] = cats
    # with open("dataset/train/train_categories.json", "w") as js:
    #     js.write(json.dumps(dct))

create_masks(split="val", ir=True)
create_masks(split="val")

create_masks(split="train", ir=True)
create_masks(split="train")
