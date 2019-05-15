#For testing dont give batch size
from network import create_model
import os
import cv2
import glob

MAIN_DIR = os.getcwd()
TESTING_DATA_FRAMES_PATH = MAIN_DIR + "/data/ILSVR2015/Data/test/*/*.JPEG"
def main():

    test_model = create_model()
    test_model.load_weights('cnn_rnn_weights.hk5')
    files = sorted(glob.glob(TESTING_DATA_FRAMES_PATH))
    for i in range(1,len(files)):
      X = np.empty((1,1,240,320, 3))
      image = cv2.imread(files[i])
      for j in range(1):
        X[0,j,] = cv2.imread(files[i-1+j])
      bbox = test_model.predict(X)
      bbox = np.int32(bbox[0])
      cv2.rectangle(image,(bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
      cv2.imwrite(MAIN_DIR+"/predictions/{0}.jpg".format(i),image)


if __name__ == '__main__':
    main()
