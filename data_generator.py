import os
import numpy as np
import keras
import glob
import cv2

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, list_id, folder_start_pos, batch_size=32, dim=(2, 240, 320), n_channels=3, shuffle=True):
        self.list_id = list_id
        self.folder_start_pos = folder_start_pos
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.path = path
        self.train_data_path =self.path+'/data2/LITIV_dataset/Data/train/'
        self.train_annot_path = self.path+'/data2/LITIV_dataset/Annotations/train/'
        self.val_data_path = self.path+'/data2/LITIV_dataset/Data/val/'
        self.val_annot_path = self.path+'/data2/LITIV_dataset/Annotations/val/'
        # self.label_list = self.get_labels()
        self.on_epoch_end()
        self.dim = dim
        self.n_channels = n_channels

    def __len__(self):
        # Denotes number of batches that make one epoch
        return int(np.floor(len(self.list_id) / self.batch_size))

    def __getitem__(self, index=1):
        # print(index)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print(indexes)
        X, Y = self.__data_generation(indexes)
        return X, Y

    def __data_generation(self, indexes):
        # print("In data generation")
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, 4), dtype=int)

        folder = [dI for dI in os.listdir(self.train_data_path) if
                  os.path.isdir(os.path.join(self.train_data_path, dI))]
        # print(folder)
        # print(len(indexes))
        img_path_list = []
        for j, fileID in enumerate(indexes):

            # Find folder index and file index
            folder_index = 0
            file_index = 0

            for i in range(len(self.folder_start_pos)):
                if (fileID < (self.folder_start_pos[i])):
                    folder_index = i
                    # print(folder_index)
                    break
            if (folder_index == 0):
                file_index = fileID
            else:
                file_index = fileID - self.folder_start_pos[folder_index - 1]

            # print("******")
            for train_len in range(2):
                # After finding in which folder the file lives, do imread of that file

                # print(file_index+train_len)
                folder_name = folder[folder_index]
                file_name = "{:04d}".format(file_index + 1 + train_len)

                img_path = self.train_data_path + folder_name + "/" + file_name + ".jpg"
                # print(img_path)
                # img_path_list.append(img_path)
                # print(len(img_path_list))
                #     print("**********")
                #     print(img_path)
                # print(fileID,folder_index, file_index+train_len)
                #     # #
                img = cv2.imread(img_path)
                X[j, train_len,] = img / 255.0
                # bbox = self.get_labels(folder_name, file_index+train_len)
                # print(bbox)
                # cv2.rectangle(img,(bbox[0]-int(bbox[2]/2),bbox[1]-int(bbox[3]/2)), (bbox[0]+int(bbox[2]/2), bbox[1]+int(bbox[3]/2)), (0,255,0), 2)
                # cv2.imshow('Chosen Image',img)
                # cv2.waitKey(0)
            Y[j] = self.get_labels(folder_name, file_index)
        # print(Y)
        return X, Y

    # Validating dataset
    def __data_generation_val(self, indexes):
        folder = [dI for dI in os.listdir(self.val_data_path) if os.path.isdir(os.path.join(self.val_data_path, dI))]
        # print(len(indexes))
        img_path_list = []
        for fileID in indexes:

            # Find folder index and file index
            folder_index = 0
            file_index = 0

            for k in range(len(self.folder_start_pos)):
                if (fileID < (self.folder_start_pos[i])):
                    folder_index = i
                    break
            if (folder_index == 0):
                file_index = fileID
            else:
                file_index = fileID - self.folder_start_pos[folder_index - 1]

            print("******")
            for val_len in range(2):
                # After finding in which folder the file lives, do imread of that file

                folder_name = folder[folder_index]
                file_name = "{:06d}".format(file_index + val_len)

                img_path = self.val_data_path + folder_name + "/" + file_name + ".JPEG"

                print(img_path)
                print(folder_index, file_index + val_len)

                img = cv2.imread(img_path)

                # Resizing bounding box
                # bbox = self.get_labels(folder_name, file_index+val_len)
                # print(bbox)
                # cv2.rectangle(img,(bbox[0]-int(bbox[2]/2),bbox[1]-int(bbox[3]/2)), (bbox[0]+int(bbox[2]/2), bbox[1]+int(bbox[3]/2)), (0,255,0), 2)
                # cv2.imshow('Chosen Image', img)
                # cv2.waitKey(0)

    def get_labels(self, folder_name, file_id):
        #      print("Get labels is called")

        labels = []
        for i in sorted(glob.glob(self.train_annot_path+folder_name+"/*.txt")):
            labels = np.loadtxt(i, delimiter=',',dtype=int)
        return labels[file_id,:]

    def on_epoch_end(self):
        self.indexes = np.arange(0, len(self.list_id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def print(self):
        X, Y = self.__getitem__(1)
        # print("Number of batches possible %d"%(self.__len__()))
        return X, Y

