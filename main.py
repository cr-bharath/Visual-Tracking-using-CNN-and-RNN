import os
import numpy as np
from data_generator import DataGenerator
from network import create_model
import keras

MAIN_DIR = os.getcwd()
# Update to training data path 
TRAIN_DATA_PATH = MAIN_DIR + '/data/ILSVR2015/Data/train/'

def prepare_for_loader():
    total_images = 0
    folder_start_pos = []
    training_data_path = TRAIN_DATA_PATH

    for _, dirnames, filenames in os.walk(training_data_path):
        if (len(filenames) != 0):
            n_files = len(filenames)
            total_images += (n_files - 1)
            folder_start_pos.append(total_images)

    #print("Folder start positions")
    #print(folder_start_pos)

    list_id = np.arange(0, total_images)
    return  list_id, folder_start_pos

def main():
    batch_size = 16

    # Prepare for creating the Batch Generator
    list_id, folder_start_pos = prepare_for_loader()
    
    # Create the Batch Generator for training Data
    trainObj = DataGenerator(MAIN_DIR, list_id, folder_start_pos,
                             batch_size,shuffle=True)
    
    model = create_model(batch_size)

    adam = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy'])
    
    # Fit the model
    model.fit_generator(generator=trainObj, epochs=50, verbose=2, use_multiprocessing=True)
    
    # Save the weights for the model
    model.save_weights('cnn_rnn_weights.hk5')


if __name__ == '__main__':
    main()


