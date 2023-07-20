# -*- coding:utf-8 -*-

import os


class ImageRename():
    def __init__(self):
        self.path = r"/home/sys120-1/cy/nir_much/trainA"
        self.path1 = r"/home/sys120-1/cy/nir_much/trainA"

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 0

        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                # dst = os.path.join(os.path.abspath(self.path), 'l' + format(str(i), '0>4s') + '.jpg')
                dst = os.path.join(os.path.abspath(self.path1), format(str(i), '0>5s') + '_A.jpg')
                os.rename(src, dst)

                print('converting %s to %s ...' % (src, dst))

                i = i + 1
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()
