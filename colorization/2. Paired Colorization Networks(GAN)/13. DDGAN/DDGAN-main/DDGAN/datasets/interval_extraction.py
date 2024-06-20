import os
import shutil
import tqdm

old_path = r'E:\colorization\init\train22/'
new_path = r'E:\colorization\ticc\trainB/'

files = os.listdir(old_path)
# for i in tqdm.tqdm(range(len(files))):
#     print(files[i][:-6])
for i in tqdm.tqdm(range(len(files))):
    if int(files[i][:-6]) % 3 == 0:
        old_file_path = old_path + '/' + files[i]
        new_file_path = new_path + '/' + files[i]
        shutil.copy(old_file_path, new_file_path)
