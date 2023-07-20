import os,random,time,math
from datetime import timedelta
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input, merge, Concatenate
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D
from keras.layers.normalization import *
from keras.optimizers import *
import matplotlib.pyplot as plt
from keras.models import Model, Sequential, load_model, model_from_json
from tqdm import tqdm
from keras import applications
from collections import OrderedDict

from keras.datasets import mnist
from utils import store_results, make_trainable, initialize_filelist, DSSIMObjective, update_dictionary, generate_image_data, export_predicted_validation_images, initialize_validation, initialize_dataset, get_soft_encoding, lab_rescale, DSSIM_L2, DSSIM_Gray, L1_DSSIM_Gray, L_DSSIM_ab_L1, load_Isola2016_generator, load_Isola2016_unet_generator, Edge_loss, L1_DSSIM_Gray_Edge, L1_DSSIM_Gray_Total_Variation, load_Cao2017_generator


#KAIST dataset
useKaist = True

#Architecture
useIsolaUnetModel_Lab = True

#Parameters
typeOfTest = 'l1_dssim_lab_day_isola'

imWidth = 640
imHeight = 512
scaleFactor = 0.5 #Image scale factor
dropoutRate = 0.25

lr = [1e-3, 1e-3]
opt_L = Adam(lr=lr[0])
opt_ab = Adam(lr=lr[0])

epochs = [100,0] #The array enables different parameters for different sets of epochs
batchSize = 8
nTestSamples = 2000

#Training
useOnlyDaySamples = True
useOnlyNightSamples = False
useBothDayAndNightSamples = False
validateOnly = False
generateValidationDataAtEveryEpoch = True

useEdgeLoss = False #I didn't try this in a while, the code is probably outdated

predict_file_list = initialize_validation(useOnlyDaySamples, useOnlyNightSamples, useBothDayAndNightSamples)

if not validateOnly:
    
	#Build model ---------------------------------------------------------------
    if useIsolaUnetModel_Lab:
        if useEdgeLoss:
            transformer_Lab = load_Isola2016_unet_generator([int(imHeight*scaleFactor), int(imWidth*scaleFactor),1], True, batchSize)
            transformer_Lab.compile(loss=L1_DSSIM_Gray_Edge(), optimizer=opt_ab, metrics=['accuracy'])
        else:
            transformer_Lab = load_Isola2016_unet_generator([int(imHeight*scaleFactor), int(imWidth*scaleFactor),1], False, batchSize)
            transformer_Lab.compile(loss=L_DSSIM_ab_L1(), optimizer=opt_ab, metrics=['accuracy'])
        transformer_Lab.summary()
    
	#You could add other architectures here

    # Initialize dataset ---------------------------------------------------------------
    datapath = 'ADD YOUR PATH HERE'
    dataset = initialize_dataset(datapath, useKaist, useOnlyDaySamples, useOnlyNightSamples, useBothDayAndNightSamples)
    nTrainSamples = int(len(dataset['LWIRtrainDataFiles'])/4)*16#OBSOBSOBS!!! This is where division by 4 happens, remove that if you want to train on the full dataset
    loss_and_accuracy = {}
    params = {}

    # Set params for result storage ----------------------------------------------------
    params = OrderedDict([(typeOfTest, ''),('ep', epochs),('bs', batchSize),('#samp',nTrainSamples),('lr',lr)])

    # Define training procedure --------------------------------------------------------
    def train_for_n(nb_epoch, steps_per_epoch, batch_size, val_size, time_per_epoch, lr_idx):

        loss_and_accuracy_e = {'transformer_Lab': {}}
        epoch_losses = {}

        for e in tqdm(range(nb_epoch)):
            epoch_time_start = time.clock()

            for step in tqdm(range(0, steps_per_epoch)):

			    #Load data
                div = 2 #The data is stored in batches of 16 and I want to use batch size 8
                remainder = step - div*math.floor(step/div)

                LWIR_train_batch = np.load(dataset['LWIRtrainDataFiles'][math.floor(step/div)])[remainder*batch_size:(remainder + 1) * batch_size,:,:,:]
                RGB_train_batch = np.load(dataset['RGBtrainDataFiles'][math.floor(step/div)])[remainder*batch_size:(remainder + 1) * batch_size,:,:,:]                
                
                #Prepare input
                X_train_batch = LWIR_train_batch / 255.0

                #Prepare fround truth
                Y_train_batch = lab_rescale(RGB_train_batch, to100=False)
                
				#Train on batch
                if useEdgeLoss
                    batch_losses = dict(zip(transformer_Lab.metrics_names, transformer_Lab.train_on_batch(X_train_batch, np.concatenate([Y_train_batch, X_train_batch], axis=3))))
                else:
                    batch_losses = dict(zip(transformer_Lab.metrics_names, transformer_Lab.train_on_batch(X_train_batch, Y_train_batch)))
                update_dictionary(epoch_losses, batch_losses)


            for key in epoch_losses.keys():
                epoch_losses[key] = epoch_losses[key].mean()
            
            update_dictionary(loss_and_accuracy_e['transformer_Lab'], epoch_losses)

            epoch_losses = {}
            total_epoch_time_elapsed = str(timedelta(seconds=(time.clock() - epoch_time_start)))
            time_per_epoch.append(total_epoch_time_elapsed)

            #For each epoch, store intermediate results
            store_path = store_results('LWIR_to_RGB', loss_and_accuracy_e, params, {'transformer_Lab': transformer_Lab}, str(timedelta(seconds=(time.clock() - time_start))), time_per_epoch, intermediate_results = True)

            if generateValidationDataAtEveryEpoch:
                X_validation_in = generate_image_data(predict_file_list, scaleFactor, imWidth, imHeight, normalize_to_interval_01 = True, 
                    convert_to_grayscale = False, reduce_to_one_channel = True)
                curr_epoch = e
                if lr_idx - 1 >= 0:
                    for ie in range(lr_idx-1):
                        curr_epoch = curr_epoch + epochs[ie]
                
				if useEdgeLoss:
					export_predicted_validation_images(X_validation_in[0:8,:,:,:], transformer_Lab, store_path, convertToLab = True, epoch = str(curr_epoch), edge = useEdgeLoss)
					export_predicted_validation_images(X_validation_in[8:16,:,:,:], transformer_Lab, store_path, convertToLab = True, epoch = str(curr_epoch), edge = useEdgeLoss, idx_offset = 8)
				else:
					export_predicted_validation_images(X_validation_in, transformer_Lab, store_path, convertToLab = False, epoch = str(curr_epoch), edge = useEdgeLoss) #OBSOBSOBS
                    
        return loss_and_accuracy_e

	# End train_for_n --------------------------------------------------------
    
	# Train network!  --------------------------------------------------------
	time_start = time.clock()
    time_per_epoch = []

    for lr_idx in range(len(lr)):
        opt_L.lr = lr[lr_idx]
        opt_ab.lr = lr[lr_idx]
        loss_and_accuracy_lr = train_for_n(epochs[lr_idx], int(nTrainSamples/batchSize), batchSize, nTestSamples, time_per_epoch, lr_idx)   
        update_dictionary(loss_and_accuracy, loss_and_accuracy_lr, dictInDict = True)

    total_time_elapsed = str(timedelta(seconds=(time.clock() - time_start)))
    store_path = store_results('LWIR_to_RGB', loss_and_accuracy, params, {'transformer_Lab': transformer_Lab}, total_time_elapsed, time_per_epoch)
	
	# Validate on some example images --------------------------------------------------------
    X_validation_in = generate_image_data(predict_file_list, scaleFactor, imWidth, imHeight, normalize_to_interval_01 = True, 
        convert_to_grayscale = False, reduce_to_one_channel = True)
	export_predicted_validation_images(X_validation_in, transformer_Lab, store_path, convertToLab = True)

else:
    X_validation_in = generate_image_data(predict_file_list, scaleFactor, imWidth, imHeight, normalize_to_interval_01 = True, 
        convert_to_grayscale = False, reduce_to_one_channel = False)
    model_path = ''
    model = load_model(model_path + 'model.h5', custom_objects={'DSSIMObjective': DSSIMObjective()})
    export_predicted_validation_images(X_validation_in, model, model_path, convertToYcbcr)