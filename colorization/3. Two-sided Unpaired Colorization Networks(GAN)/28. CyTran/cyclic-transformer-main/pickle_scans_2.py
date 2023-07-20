import os
import pickle
import pydicom


# dont forget you might have to add the preprocessing here


# set directory paths
dir_A = 'Coltea-Lung-CT-100W/train/trainA/'
dir_B = 'Coltea-Lung-CT-100W/train/trainB/'

# set output
output_file = 'Coltea-Lung-CT-100W/train/train_data.pkl'

# initialize dataset dictionary
data = {'A': [], 'B': []}

# iterate over images in dir_A
for file_name in os.listdir(dir_A):
    # open image file and append to list for data['A']
    # img = pydicom.dcmread(os.path.join(dir_A, file_name))
    data['A'].append(os.path.join(dir_A, file_name))

# iterate over images in dir_B
for file_name in os.listdir(dir_B):
    # open image file and append to list for data['B']
    # img = pydicom.dcmread(os.path.join(dir_B, file_name))
    data['B'].append(os.path.join(dir_B, file_name))

# save pickled dataset to output_file
with open(output_file, 'wb') as f:
    pickle.dump(data, f)