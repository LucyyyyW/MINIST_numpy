import numpy as np
from PIL import Image
import os
import struct
import matplotlib.pyplot as plt

class ImageLoader():
    def __init__(self, img_path, lab_path, batch_size):
        self.images = self.decode_image(img_path)
        self.labels = self.decode_label(lab_path)
        self.batch_size = batch_size
        self.len = 0

    def __iter__(self):
        self.id = 0
        return self
    
    #read images
    def decode_image(self,idx3_ubyte_file):
        '''
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        '''
        # 读取二进制数据
        bin_data = open(idx3_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset = 0
        fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
        self.len = num_images

        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
        print(offset)
        fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
        print(fmt_image,offset,struct.calcsize(fmt_image))
        images = np.empty((num_images, num_rows, num_cols))
        # plt.figure()
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('已解析 %d' % (i + 1) + '张')
                print(offset)
            images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
            #print(images[i])
            offset += struct.calcsize(fmt_image)
        #     plt.imshow(images[i],'gray')
        #     plt.pause(0.00001)
        #     plt.show()
        # plt.show()
        return images
    
    #read label
    def decode_label(self,idx1_ubyte_file):
        """
        解析idx1文件的通用函数
        :param idx1_ubyte_file: idx1文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx1_ubyte_file, 'rb').read()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print ('已解析 %d' % (i + 1) + '张')
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels
        
    def __next__(self):
        imgs = []
        labs = []
        for i in range(self.batch_size):
            lab = np.zeros(10)
            imgs.append(self.images[self.id+i].flatten())
            lab[int(self.labels[self.id+i])] = 1
            labs.append(lab)
        imgs = np.stack(imgs,0)
        labs = np.stack(labs,0)
        self.id += self.batch_size
        return imgs, labs
    
    def __len__(self):
        return self.len

# img_path = '/Users/wsh/Downloads/t10k-images.idx3-ubyte'
# lab_path = '/Users/wsh/Downloads/t10k-labels.idx1-ubyte'
# a = ImageLoader(img_path, lab_path)
# myiter = iter(a)

# img, lab = next(myiter)
# print(lab)
