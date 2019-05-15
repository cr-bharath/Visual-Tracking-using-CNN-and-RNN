from keras import backend as K
from keras.engine import Layer
from keras.layers import Conv2D
from keras.layers.core import Lambda
from keras.layers import Activation,Dense,Dropout,Flatten, Input,merge,TimeDistributed
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.models import Model,Sequential
from keras.optimizers import SGD
from keras.applications.inception_v3  import InceptionV3 as iv3
from keras.layers import Concatenate,Reshape
from keras.layers import LSTM
import keras


def create_model(batch_size=None):
  """ Function to create a CNN-LSTM model for visual tracking"""
  K.set_learning_phase(1)
  # You can replace iv3 with any other ImageNet pretrained models from Keras and compare performance of all
  cnn = iv3(
    weights='imagenet',
    include_top=False,
    pooling='max')
  
  if batch_size != None:
    video = Input(shape=(2,240, 320, 3), name='video_input')
    encoded_frame = TimeDistributed(cnn)(video)
    encoded_vid = LSTM(1024,unroll=True)(encoded_frame)
  else:
    # This is to create model while testing to set stateful=True
    video = Input(batch_shape=( 1,1,240, 320, 3),name='video_input')
    encoded_frame = TimeDistributed(cnn)(video)
    encoded_vid = LSTM(1024,unroll=False,stateful=True)(encoded_frame)

  outputs = Dense(4, activation=None,kernel_regularizer=keras.regularizers.l2(0.01),
                activity_regularizer=keras.regularizers.l1(0.01))(encoded_vid)
  model = Model(inputs=[video],outputs=outputs)
  return model
