import os
import cv2
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util.visualizer import save_images2
from util import html
from PIL import Image
import torchvision.transforms as transforms


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    #data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()

    model = create_model(opt)
    model.setup(opt)

    i = 0
    # test
    while True:
        images = []
        for root, _, fnames in sorted(os.walk("datasets/grad_gray/test1/")):
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)

        A = Image.open(images[0]).convert('RGB')
        A = transforms.ToTensor()(A)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        A = A.unsqueeze(0)
	#print(A)  
	
        model.set_input(A)
        model.test()
        visuals = model.get_current_visuals()
        #img_path = model.get_image_paths()
        #if i % 5 == 0:
            #print('processing (%04d)-th image... %s' % (i, img_path))
        #save_images2(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
	i = i + 1
       
        
        fileDelPath=images[0]
        fileDelName=os.path.basename(fileDelPath)
        print(fileDelName)
        src=cv2.imread(fileDelPath)
        cv2.namedWindow("test",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("test",src)
        os.remove(fileDelPath)
        cv2.waitKey(50)
        #os.path.exists(fileDelPath)
        #if not os.path.exists(fileDelPath):
           #break
        #cv2.destroyAllWindows()
       


