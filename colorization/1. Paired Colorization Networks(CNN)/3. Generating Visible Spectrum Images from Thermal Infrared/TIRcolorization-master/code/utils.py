import os, sys
import csv
import datetime
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#import theano
#import theano.tensor as T
import tensorflow as TF
#from theano.sandbox.neighbours import images2neibs
import keras.backend as K
#from keras.backend import theano_backend as KTH
from keras.backend import tensorflow_backend as KTF
from skimage import color
import sklearn.neighbors as nn
from scipy.stats import multivariate_normal
from skimage import filters

from keras.layers import Input, Merge, Concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model, model_from_json
from keras.losses import mean_absolute_error, mean_squared_error
from keras.optimizers import *

def write_dict_to_csv(file, dict):
    with open(file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict.items():
            writer.writerow([key, value])

def read_csv_to_dict(file):
    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        return dict(reader)

def visualize_lab_validation(Y_pred, L, batch_size, h, w, nb_q, q_ab, T):

    # Format X_colorized
    X_colorized = Y_pred[:, :, :, :-1]
    X_colorized = X_colorized.reshape((batch_size * h * w, nb_q))

    # Reweight probas
    X_colorized = np.exp(np.log(X_colorized) / T)
    X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

    # Reweighted
    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    X_a = np.sum(X_colorized * q_a, 1).reshape((batch_size, 1, h, w))
    X_b = np.sum(X_colorized * q_b, 1).reshape((batch_size, 1, h, w))
    L = np.expand_dims(L, axis = 1)
    X_colorized = np.concatenate((L, X_a, X_b), axis=1).transpose(0, 2, 3, 1)
    X_colorized = [np.expand_dims(color.lab2rgb(im), 0) for im in X_colorized]

    return X_colorized


def export_predicted_validation_images(X_in, model, storepath, convertToYcbcr = False, convertToLab = False, useLfromRGB = False, epoch = '', edge = False, idx_offset = 0):
    Y = model.predict(X_in)
    if edge:
        Y = Y[:,:,:,0:3]
    validationPath = storepath + 'validation' + epoch
    if not os.path.exists(validationPath):
        os.makedirs(validationPath) 

    if convertToLab and useLfromRGB:
        #Temp test where the Lab component comes from the RGB image
        validationpath = 'ADD YOUR PATH HERE'
        predict_file_list = []
        for root, dirs, files in os.walk(validationpath):
            for name in files:
                if name.endswith((".png")) and (root.find("lwir") >= 0):
                    predict_file_list.append(root + "/" + name)
        LWIR_data = generate_image_lab_data(predict_file_list, 0.5, 640, 512)[0:4,:,:,:]
        folder = 'ADD YOUR PATH HERE'
        with open(folder + 'transformer_Lab.json') as f:
            json_string = f.readline()
        transformer_L = model_from_json(json_string)
        transformer_L.load_weights(folder + 'transformer_Lab.h5')
        lr = [1e-3, 1e-3]
        opt_L = Adam(lr=lr[0])
        transformer_L.compile(loss=DSSIMObjective(), optimizer=opt_L, metrics=['accuracy'])
        Y_true = lab_rescale(transformer_L.predict_on_batch(np.expand_dims(LWIR_data[:,:,:,0]/255, axis=-1)), to100=True)
        q_ab = np.load("ADD YOUR PATH HERE/pts_in_hull.npy")
        nb_q = q_ab.shape[0]
        Y = visualize_lab_validation(Y, Y_true[:,:,:,0], 4, 256, 320, nb_q, q_ab, 0.38)

    if convertToLab and not useLfromRGB:
        Y = lab_rescale(Y, to100 = True)

    for idx, image in enumerate(Y):
        if convertToYcbcr:
            image = image * 255
            image = ycbcr_to_rgb(image)
        if convertToLab and not useLfromRGB:
            image = color.lab2rgb(image.astype('float64'))
        
        image = image*255
        im = Image.fromarray(np.squeeze(image).astype('uint8'))
        im.save(validationPath + '/I000' + str(idx+idx_offset) + '.png')

def initialize_validation(useOnlyDaySamples, useOnlyNightSamples, useBothDayAndNightSamples):
    #Validation
    if useOnlyDaySamples:
        validationDirectory = 'ADD YOUR PATH HERE/Day/'
    if useOnlyNightSamples:
        validationDirectory = 'ADD YOUR PATH HERE/Night/'
    if useBothDayAndNightSamples:
        validationDirectory = 'ADD YOUR PATH HERE'

    predict_file_list = []
    for root, dirs, files in os.walk(validationDirectory):
        for name in files:
            if name.endswith((".png")) and (root.find("lwir") >= 0):
                predict_file_list.append(root + "/" + name)

    return predict_file_list

def initialize_dataset(datapath, useKaist, useOnlyDaySamples, useOnlyNightSamples, useBothDayAndNightSamples):
    if useKaist:
        if useOnlyDaySamples:
            LWIRtrainDataFiles = initialize_filelist(datapath + 'train/day/lwir/', filending = '.npy', findstr = '')
            RGBtrainDataFiles = initialize_filelist(datapath + 'train/day/visual/', filending = '.npy', findstr = '')
            LWIRtestDataFiles = initialize_filelist(datapath + 'test/day/lwir/', filending = '.npy', findstr = '')
            RGBtestDataFiles = initialize_filelist(datapath + 'test/day/visual/', filending = '.npy', findstr = '')
        if useOnlyNightSamples:
            LWIRtrainDataFiles = initialize_filelist(datapath + 'train/night/lwir/', filending = '.npy', findstr = '')
            RGBtrainDataFiles = initialize_filelist(datapath + 'train/night/visual/', filending = '.npy', findstr = '')
            LWIRtestDataFiles = initialize_filelist(datapath + 'test/night/lwir/', filending = '.npy', findstr = '')
            RGBtestDataFiles = initialize_filelist(datapath + 'test/night/visual/', filending = '.npy', findstr = '')
        if useBothDayAndNightSamples:
            #Not implemented yet
    
    dataset = {'LWIRtrainDataFiles': LWIRtrainDataFiles, 'RGBtrainDataFiles': RGBtrainDataFiles,
                'LWIRtestDataFiles': LWIRtestDataFiles, 'RGBtestDataFiles': RGBtestDataFiles}    

    return dataset

#losses = {"subnetwork1":[], "subnetwork2":[], ....}
#accuracies = {"subnetwork1":[], "subnetwork2":[], ....}
def store_results(type_of_test, loss_and_acc, params, models, total_execution_time, execution_times_per_epoch, intermediate_results=False):#model, model_parameters, losses, accuracies):
    
    #Generate folder name from parameters
    dtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = type_of_test + '/'
    for param in params.keys():
        folder_name = folder_name + param + '_' + str(params[param]) + '_'
    
    store_test_dir = 'ADD YOUR PATH HERE' + folder_name + '/'
    
    if intermediate_results:
        store_test_dir = store_test_dir + '/tmp/'
    else:
        store_test_dir = store_test_dir + dtime + '/'

    if not os.path.exists(store_test_dir):
        os.makedirs(store_test_dir)
    
    #Store model
    for key in models:
        #models[key].save(store_test_dir + key + '.h5')
        model_json =models[key].to_yaml()
        with open(store_test_dir+key+'.yaml', 'w') as json_file:
            json_file.write(model_json)
        models[key].save_weights(store_test_dir+key+'.h5')
    
    #Store losses and accuracies in file and plot them 
    write_dict_to_csv(store_test_dir + 'losses_and_acc.csv', loss_and_acc)
    
    for key in loss_and_acc:
        for metric in loss_and_acc[key]:
            if loss_and_acc[key][metric].size > 0:
                plt.clf()
                plt.plot(loss_and_acc[key][metric])
                plt.title(key + '_' + metric)
                plt.ylabel(metric)
                plt.xlabel('epoch')
                plt.savefig(store_test_dir + key + '_' + metric + '_plot.png')
    
    #Store parameters in file
    write_dict_to_csv(store_test_dir + 'parameters.csv', params)

    #Write model summary to file
    for key in models:
        orig_stdout = sys.stdout
        f = open(store_test_dir + key + '_summary.txt', 'w')
        sys.stdout = f
        print(models[key].summary())
        sys.stdout = orig_stdout
        f.close()
    
    #Write execution times
    f = open(store_test_dir + 'execution_times.txt', 'w')
    f.write('Total execution time: ' + total_execution_time + '\n\n')
    f.write('Execution time per epoch\n')
    f.write('------------------------------------------\n')
    for idx, time in enumerate(execution_times_per_epoch): f.write(str(idx) + ': ' + time + '\n')
    f.close()
    
    return store_test_dir


def update_dictionary(orig, new, dictInDict = False):
    for key in new.keys():
        if key in orig.keys():
            if dictInDict:
                for key2 in new[key].keys():
                    if key2 in orig[key].keys():
                        orig[key][key2] = np.append(orig[key][key2], new[key][key2])
                    else:
                        orig[key][key2] = new[key][key2]
            else:
                orig[key] = np.append(orig[key], new[key])
        else:
            orig[key] = new[key]
    return orig


# Data generation functions--------------------------------------------------------------

def generate_patch_data(file_list, patch_width, patch_height, normalize_to_interval_01 = True, convert_to_grayscale = False):
    #Randomly select a patch and cut
    patches = []
    for file in file_list:
        if convert_to_grayscale:
            im = np.expand_dims(np.array(Image.open(file).convert("L")), axis = 2)
        else:
            im = np.expand_dims(np.array(Image.open(file))[:,:,0], axis = 2)
        startIdxX = np.random.randint(0, im.shape[1]-patch_width, 1)
        startIdxY = np.random.randint(0, im.shape[0]-patch_height, 1)
        cropped_im = im[startIdxY:startIdxY+patch_height, startIdxX:startIdxX+patch_width, :]
        if normalize_to_interval_01:
            cropped_im = cropped_im.astype('float32') / 255
        patches.append(cropped_im)
    return patches

def generate_patch_data_no_edges(file_list, patch_width, patch_height, normalize_to_interval_01 = True, convert_to_grayscale = False):
    #Randomly select a patch and cut
    margin = 50;
    patches = []
    for file in file_list:
        if convert_to_grayscale:
            im = np.expand_dims(np.array(Image.open(file).convert("L")), axis = 2)
        else:
            im = np.expand_dims(np.array(Image.open(file))[:,:,0], axis = 2)
        startIdxX = np.random.randint(margin, im.shape[1]-patch_width-margin, 1)
        startIdxY = np.random.randint(margin, im.shape[0]-patch_height-margin, 1)
        cropped_im = im[startIdxY:startIdxY+patch_height, startIdxX:startIdxX+patch_width, :]
        if normalize_to_interval_01:
            cropped_im = cropped_im.astype('float32') / 255
        patches.append(cropped_im)
    return patches

def rgb_to_lab(rgbIm):  
    return color.rgb2lab(rgbIm)

def lab_to_rgb(labIm):
    return color.lab2rgb(labIm) * 255

def lab_rescale(im, to100):
    if to100:
        im[:,:,:,0] = im[:,:,:,0]*100.0
        im[:,:,:,1] = im[:,:,:,1]*185.0 - 87.0
        im[:,:,:,2] = im[:,:,:,2]*203.0 - 108.0
        return im
    else:
        im[:,:,:,0] = im[:,:,:,0]/100.0
        im[:,:,:,1] = (im[:,:,:,1] + 87.0)/185.0
        im[:,:,:,2] = (im[:,:,:,2] + 108.0)/203.0
        return im


def rgb_to_ycbcr(rgbIm): # in (0,255) range
    result = np.zeros(rgbIm.shape)
    result[:,:,0] = .299*rgbIm[:,:,0] + .587*rgbIm[:,:,1] + .114*rgbIm[:,:,2]
    result[:,:,1] = 128 -.168736*rgbIm[:,:,0] -.331364*rgbIm[:,:,1] + .5*rgbIm[:,:,2]
    result[:,:,2] = 128 +.5*rgbIm[:,:,0] - .418688*rgbIm[:,:,1] - .081312*rgbIm[:,:,2]
    return result

def ycbcr_to_rgb(ycbcrIm):
    result = np.zeros(ycbcrIm.shape)
    result[:,:,0] = ycbcrIm[:,:,0] + 1.402 * (ycbcrIm[:,:,2]-128)
    result[:,:,1] = ycbcrIm[:,:,0] - .344136 * (ycbcrIm[:,:,1]-128) -  .714136 * (ycbcrIm[:,:,2]-128)
    result[:,:,2] = ycbcrIm[:,:,0] + 1.772 * (ycbcrIm[:,:,1]-128)
    return result

def make_trainable(net, val, vgg = False):
    net.trainable = val
    for l in net.layers:
        if vgg:
            if l >= 19:
                l.trainable = val
        else:
            l.trainable = val

def initialize_filelist(path, filending = '.png', findstr = 'lwir'):
    predict_file_list = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((filending)) and (root.find(findstr) >= 0):
                predict_file_list.append(root + "/" + name)
    return predict_file_list

def generate_image_lab_data(file_list, scaleFactor, imWidth, imHeight, normalize_to_interval_01 = False, quantizeAccToZhang2016 = False, 
    nn_finder = (), nb_q = 0, prior_factor = []):
    images = np.array([rgb_to_lab(np.array(
            Image.open(im).resize((int(imWidth*scaleFactor), int(imHeight*scaleFactor))))) 
                for im in file_list], 'f') 
    if normalize_to_interval_01:
        images[:,:,:,0] = images[:,:,:,0] .astype('float32')/100
        images[:,:,:,1] = (images[:,:,:,1] .astype('float32')+87)/185
        images[:,:,:,2] = (images[:,:,:,2] .astype('float32')+108)/203
    #https://stackoverflow.com/questions/19099063/what-are-the-ranges-of-coordinates-in-the-cielab-color-space

    if quantizeAccToZhang2016:
        X_batch_ab = images[:, :, :, 1:]
        npts, h, w, c = X_batch_ab.shape
        X_a = np.ravel(X_batch_ab[:, :, :, 0])
        X_b = np.ravel(X_batch_ab[:, :, :, 1])
        X_batch_ab = np.vstack((X_a, X_b)).T
        Y_batch = get_soft_encoding(X_batch_ab, nn_finder, nb_q)
        idx_max = np.argmax(Y_batch, axis=1)
        weights = prior_factor[idx_max].reshape(Y_batch.shape[0], 1)
        Y_batch = np.concatenate((Y_batch, weights), axis=1)
        # # Reshape Y_batch
        images = Y_batch.reshape((npts, h, w, nb_q + 1))

    return images

def generate_image_data(file_list, scaleFactor, imWidth, imHeight, normalize_to_interval_01 = True, 
convert_to_grayscale = False, reduce_to_one_channel = False, convert_to_ycbcr = False):
    if convert_to_grayscale:
        images = np.array([np.expand_dims(np.array(
            Image.open(im).resize((int(imWidth*scaleFactor), int(imHeight*scaleFactor))).convert("L")), axis = 2) 
                for im in file_list], 'f')
    else:
        if convert_to_ycbcr:
            images = np.array([rgb_to_ycbcr(np.array(
            Image.open(im).resize((int(imWidth*scaleFactor), int(imHeight*scaleFactor))))) 
                for im in file_list], 'f')
        else:
            if reduce_to_one_channel:
                images = np.array([np.expand_dims(np.array(
                    Image.open(im).resize((int(imWidth*scaleFactor), int(imHeight*scaleFactor))))[:,:,0], axis = 2) 
                        for im in file_list], 'f')
            else: 
                images = np.array([np.array(
                    Image.open(im).resize((int(imWidth*scaleFactor), int(imHeight*scaleFactor))))
                        for im in file_list], 'f')
    if normalize_to_interval_01:
        images = images.astype('float32')/255
    return images


# Loss functions ---------------------------------------------
epsilon = 1.0e-9

def extract_image_patches(X, ksizes, strides, rates, padding='valid', data_format='channels_first'):
    '''
    Extract the patches from an image
    Parameters
    ----------
    X : The input image
    ksizes : 2-d tuple with the kernel size
    strides : 2-d tuple with the strides size
    padding : 'same' or 'valid'
    data_format : 'channels_last' or 'channels_first'
    Returns
    -------
    The (k_w,k_h) patches extracted
    TF ==> (batch_size,w,h,k_w,k_h,c)
    TH ==> (batch_size,w,h,c,k_w,k_h)
    https://github.com/farizrahman4u/keras-contrib/blob/master/keras_contrib/backend/theano_backend.py
    '''
    if padding == 'same':
        padding = 'ignore_borders'
    if data_format == 'channels_first':
        X = KTF.permute_dimensions(X, [0, 2, 3, 1])
    # Thanks to https://github.com/awentzonline for the help!
    patches = TF.extract_image_patches(X, ksizes, strides, rates, padding.upper())
    return patches

class DSSIMObjective():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=[4,16], max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
            https://github.com/farizrahman4u/keras-contrib/blob/master/keras_contrib/losses/dssim.py
        """
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [1, self.kernel_size[0], self.kernel_size[1], 1]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = extract_image_patches(y_pred, kernel, kernel, [1,1,1,1], 'valid', self.dim_ordering)
        patches_true = extract_image_patches(y_true, kernel, kernel, [1,1,1,1], 'valid', self.dim_ordering)

        # Reshape to get the var in the cells
        #bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        #patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        #patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)

class SSIM():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=[4,16], max_value=1.0):
        self.__name__ = 'SSIM'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [1, self.kernel_size[0], self.kernel_size[1], 1]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = extract_image_patches(y_pred, kernel, kernel, [1,1,1,1], 'valid', self.dim_ordering)
        patches_true = extract_image_patches(y_true, kernel, kernel, [1,1,1,1], 'valid', self.dim_ordering)

        # Reshape to get the var in the cells
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean(ssim)

class DSSIM_L1():

    def __init__(self, alpha=0.5, beta=0.5):

        self.__name__ = 'DSSIM_L1'
        self.__alpha = alpha
        self.__beta = beta   
        self.__dssim = DSSIMObjective()

    def __call__(self, y_true, y_pred):
        alpha = 0.5
        beta = 0.5
        return alpha*mean_absolute_error(y_true, y_pred) + beta * DSSIMObjective.__call__(self.__dssim,y_true,y_pred)

class DSSIM_L2():

    def __init__(self, alpha=0.5, beta=0.5):

        self.__name__ = 'DSSIM_L2'
        self.__alpha = alpha
        self.__beta = beta   
        self.__dssim = DSSIMObjective()

    def __call__(self, y_true, y_pred):
        return self.__alpha*mean_squared_error(y_true, y_pred) + self.__beta * DSSIMObjective.__call__(self.__dssim,y_true,y_pred)

class Gray_loss():
    def __init__(self, m = 0.0, sigma = 0.25):
        self.__name__ = 'Gray_loss'
        self.__m = m
        self.__sigma = sigma
        self.__dist = TF.contrib.distributions.MultivariateNormalDiag([m,m], [sigma,sigma])
        self.__p = 1/(2*math.pi*sigma**2) #(2 pi)^(-k/2) |det(C)|^(-1/2)

    def __call__(self, y_pred):
        gray_loss = self.__dist.prob(y_pred[:,:,:,1:3]) #a&b
        return K.mean(gray_loss/self.__p)#TF.reduce_sum(gray_loss)/(K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2])

class Edge_loss():
    def __init__(self):
        self.__name__ = 'Edge_loss'
        self.__sobel_x = TF.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], TF.float32)
        self.__sobel_x_filter = TF.reshape(self.__sobel_x, [3, 3, 1, 1])
        self.__sobel_y_filter = TF.transpose(self.__sobel_x_filter, [1, 0, 2, 3])

    def __filter_image__(self, image):
        filter_image = TF.expand_dims(TF.transpose(image, [1,2,0]), axis = -1)        
        filtered_x = TF.nn.conv2d(filter_image, self.__sobel_x_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
        filtered_y = TF.nn.conv2d(filter_image, self.__sobel_y_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
        filtered_x_t = TF.transpose(filtered_x, [2,0,1,3])
        filtered_y_t = TF.transpose(filtered_y, [2,0,1,3])
        return TF.concat([filtered_x_t, filtered_y_t], axis=3)

    def __call__(self, y_true, y_pred):
        edge_im_true = self.__filter_image__(y_true)#LWIR
        edge_im_pred = self.__filter_image__(y_pred)#L
        #return mean_squared_error(K.abs(edge_im_true), K.abs(edge_im_pred))
        return mean_squared_error(edge_im_true, edge_im_pred)

class L1_DSSIM_Gray_Edge():
    def __init__(self, alpha=1/3, beta=1/3, gamma=1/3, delta = 1, rgb = False):
        self.__name__ = 'L1_DSSIM_Gray_Edge_loss'
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__delta = delta
        self.__dssim = DSSIMObjective()
        self.__gray = Gray_loss()
        self.__edge = Edge_loss()
        self.__rgb = rgb

    def __call__(self, y_true, y_pred):
        if self.__rgb:
            return self.__alpha*mean_absolute_error(y_true, y_pred) + self.__beta * DSSIMObjective.__call__(self.__dssim,y_true,y_pred) + self.__gamma*Gray_loss.__call__(self.__gray, y_pred) + self.__delta*Edge_loss.__call__(self.__edge, y_true[:,:,:,0], y_pred[:,:,:,0])
        else: #OBS this is also rgb at the moment!!!
            return self.__alpha*mean_absolute_error(y_true[:,:,:,0:3], y_pred[:,:,:,0:3]) + self.__beta * DSSIMObjective.__call__(self.__dssim,y_true[:,:,:,0:3],y_pred[:,:,:,0:3]) + self.__gamma*Gray_loss.__call__(self.__gray, y_pred[:,:,:,0:3]) + self.__delta*Edge_loss.__call__(self.__edge, y_true[:,:,:,0], y_pred[:,:,:,0])


class DSSIM_Gray():
    def __init__(self, alpha=0.5, beta=0.5):
        self.__name__ = 'DSSIM_Gray_loss'
        self.__alpha = alpha
        self.__beta = beta
        self.__dssim = DSSIMObjective()
        self.__gray = Gray_loss()

    def __call__(self, y_true, y_pred):
        return self.__alpha*Gray_loss.__call__(self.__gray, y_pred) + self.__beta * DSSIMObjective.__call__(self.__dssim,y_true,y_pred)

class L1_DSSIM_Gray():
    def __init__(self, alpha=1/3, beta=1/3, gamma=1/3):
        self.__name__ = 'L1_DSSIM_Gray_loss'
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__dssim = DSSIMObjective()
        self.__gray = Gray_loss()

    def __call__(self, y_true, y_pred):
        return self.__alpha*mean_absolute_error(y_true, y_pred) + self.__beta * DSSIMObjective.__call__(self.__dssim,y_true,y_pred) + self.__gamma*Gray_loss.__call__(self.__gray, y_pred)
     

class L1_DSSIM_Gray_Total_Variation():
    def __init__(self, alpha=0.5, beta=0.5, gamma=0, lamb = 0):
        self.__name__ = 'L1_DSSIM_Gray_Total_Variation_loss'
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__lamb = lamb
        self.__dssim = DSSIMObjective()
        self.__gray = Gray_loss()

    def __call__(self, y_true, y_pred):
        return self.__alpha*mean_absolute_error(y_true[:,:,:,1:3], y_pred[:,:,:,1:3]) + self.__beta * DSSIMObjective.__call__(self.__dssim,TF.expand_dims(y_true[:,:,:,0], axis=-1),TF.expand_dims(y_pred[:,:,:,0], axis=-1)) + self.__gamma*Gray_loss.__call__(self.__gray, y_pred) + self.__lamb*TF.reduce_sum(TF.image.total_variation(y_pred))/(K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2]*K.int_shape(y_pred)[3])#total_variation return 1Dtensor with shape batch


class L_DSSIM_ab_L1():
    def __init__(self, alpha=0.5, beta=0.5):
        self.__name__ = 'L_DSSIM_ab_L1_loss'
        self.__alpha = alpha
        self.__beta = beta
        self.__dssim = DSSIMObjective()

    def __call__(self, y_true, y_pred):
        return self.__alpha*mean_absolute_error(y_true[:,:,:,1:3], y_pred[:,:,:,1:3]) + self.__beta * DSSIMObjective.__call__(self.__dssim,TF.expand_dims(y_true[:,:,:,0], axis=-1),TF.expand_dims(y_pred[:,:,:,0], axis=-1))



# Architectures ---------------------------------------------
def C_block(H, nch, bn, downsample):
    H = LeakyReLU(0.2)(H)
    if downsample:
        H = Conv2D(nch, (4,4), strides = (2,2), padding = 'same', kernel_initializer='glorot_normal')(H)
    else:
        H = Conv2D(nch, (4,4), padding = 'same', kernel_initializer='glorot_normal')(H)
    if bn:
        H = BatchNormalization(axis = 1)(H)
    return H

def CD_block(H, nch, bn, upsample):
    H = LeakyReLU(0.0)(H)
    #H = Conv2DTranspose(nch, (4,4), strides = (2,2), padding = 'same', kernel_initializer='glorot_normal')(H)
    if upsample:
        H = UpSampling2D(size = (2,2))(H)
    H = Conv2D(nch, (4,4), padding = 'same', kernel_initializer='glorot_normal')(H)
    if bn:
        H = BatchNormalization(axis = 1)(H)
    H = Dropout(0.5)(H)
    return H

def CD_skip_block(H, Hskip, nch, bn, upsample):
    H = LeakyReLU(0.0)(H)
    #H = Conv2DTranspose(nch, (4,4), strides = (2,2), padding = 'same', kernel_initializer='glorot_normal')(H)
    if upsample:
        H = UpSampling2D(size = (2,2))(H)
    H = Conv2D(nch, (4,4), padding = 'same', kernel_initializer='glorot_normal')(H)
    if bn:
        H = BatchNormalization(axis = 1)(H)
    H = Dropout(0.5)(H)
    H = Concatenate(axis = -1)([H, Hskip])    
    return H

def load_Isola2016_generator(input_shape):
    
    nch_max = 256
    m_input = Input(shape=input_shape)
    H = m_input
    
    #Encoder
    nch = [int(nch_max/8), int(nch_max/4), int(nch_max/2), int(nch_max/2)]#, nch_max]
    depth = len(nch)

    for d in range(depth):
        if d == 0:
            H = C_block(H, nch[d], False, True)
        else:
            if d < 2:
                H = C_block(H, nch[d], True, True)
            else:
                H = C_block(H, nch[d], True, False)

    #Decoder
    for d in range(depth):
        if depth-d-1 < 2:
            H = CD_block(H, nch[depth-d-1], True, True)
        else:
            H = CD_block(H, nch[depth-d-1], True, False)

    m_output = Conv2D(3, (4,4), activation = 'sigmoid', padding = 'same', kernel_initializer='glorot_normal')(H)

    return Model(m_input, m_output)

def load_Isola2016_unet_generator(input_shape, input_as_output, batch_size, estimateOnlyL=False):
    
    nch_max = 256
    m_input = Input(shape=input_shape, name='lwir_in')
    H = m_input
    
    #Encoder
    nch = [int(nch_max/8), int(nch_max/4), int(nch_max/2), int(nch_max/2)]#, nch_max]
    depth = len(nch)
    
    encoder_layers = []

    for d in range(depth):
        if d == 0:
            H = C_block(H, nch[d], False, True)
            encoder_layers.append(H)
        else:
            if d < 2:
                H = C_block(H, nch[d], True, True)
            else:
                H = C_block(H, nch[d], True, False)
            encoder_layers.append(H)

    #Decoder
    for d in range(depth):
        if d < depth-1:
            if depth-d-1 < 2:
                H = CD_skip_block(H, encoder_layers[depth-d-2], nch[depth-d-1], True, True)
            else:
                H = CD_skip_block(H, encoder_layers[depth-d-2], nch[depth-d-1], True, False)
        else:
            H = CD_block(H, nch[depth-d-1], True, True)

    if estimateOnlyL:
        m_output = Conv2D(1, (4,4), activation = 'sigmoid', padding = 'same', kernel_initializer='glorot_normal', name='lab_out_1')(H)
    else:
        m_output = Conv2D(3, (4,4), activation = 'sigmoid', padding = 'same', kernel_initializer='glorot_normal', name='lab_out_1')(H)

    if input_as_output:
        def output_shape(input_shape):
            return (input_shape[1], input_shape[2], input_shape[3]+1)

        def reshape_output(x, input_shape):
            # Add a zero layer so that x has the same dimension as the target 
            sh = K.int_shape(x)
            xc = K.zeros((batch_size, sh[1], sh[2], 1))
            x = K.concatenate([x, xc], axis=3)
            return x    

        Reshape_output = Lambda(lambda z: reshape_output(z, input_shape), output_shape=output_shape(K.int_shape(m_output)), name="ReshapeOutput")
        m_output = Reshape_output(m_output)
        out_model = Model(m_input, [m_output])
    else:
        out_model = Model(m_input, m_output)

    return out_model


def Cao_block(H, nch):
    H = Conv2D(nch, (3,3), strides = (1,1), padding = 'same', kernel_initializer='glorot_normal')(H)
    H = BatchNormalization(axis = 1)(H)
    H = LeakyReLU(0.2)(H)
    return H

def load_Cao2017_generator(input_shape):
    
    nch_max = 256
    m_input = Input(shape=input_shape)
    H = m_input
    nch = [int(nch_max/2), int(nch_max/4), int(nch_max/4), int(nch_max/4), int(nch_max/8)]
    depth = len(nch)

    for d in range(depth):
        H = Cao_block(H, nch[d])

    m_output = Conv2D(3, (3,3), activation = 'sigmoid', padding = 'same', kernel_initializer='glorot_normal')(H)

    return Model(m_input, m_output)



#-------Other stuff-----------------------------
def get_soft_encoding(X, nn_finder, nb_q):
    sigma_neighbor = 5

    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(X)

    # Smooth the weights with a gaussian kernel
    wts = np.exp(-dist_neighb**2 / (2 * sigma_neighbor**2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]

    # format the target
    Y = np.zeros((X.shape[0], nb_q))
    idx_pts = np.arange(X.shape[0])[:, np.newaxis]
    Y[idx_pts, idx_neigh] = wts

    return Y