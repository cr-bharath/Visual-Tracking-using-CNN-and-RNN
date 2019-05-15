import os
import numpy as np
from data_generator import DataGenerator
from network import create_model
import keras

def prepare_for_loader(current_path):
    total_images = 0
    folder_start_pos = []
    training_data_path = current_path + '/data/LITIV_dataset/Data/train/'

    for _, dirnames, filenames in os.walk(training_data_path):
        print(dirnames)
        if (len(filenames) != 0):
            n_files = len(filenames)
            total_images += (n_files - 1)
            folder_start_pos.append(total_images)

    print("Folder start positions")
    print(folder_start_pos)

    list_id = np.arange(0, total_images)
    return  list_id, folder_start_pos

def main():
    batch_size = 16
    current_path = os.getcwd()

    list_id, folder_start_pos = prepare_for_loader(current_path)
    trainObj = DataGenerator(current_path, list_id, folder_start_pos,
                             batch_size,shuffle=True)
    # trainObj.print()

    model = create_model(batch_size)

    adam = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy'])
    model.fit_generator(generator=trainObj, epochs=50, verbose=2, use_multiprocessing=True)

    model.save_weights('cnn_rnn_weights.hk5')


if __name__ == '__main__':
    main()


