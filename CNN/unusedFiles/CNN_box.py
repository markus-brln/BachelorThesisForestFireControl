import random
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, \
  Reshape, Activation
from NNutils import *


# tf.random.set_seed(123)
# np.random.seed(123)


def load_data(out_variant):
  print("loading data")
  images = np.load("images_" + out_variant + ".npy", allow_pickle=True)
  # concat = np.load("concat_" + out_variant + ".npy", allow_pickle=True)
  outputs = np.load("outputs_" + out_variant + ".npy", allow_pickle=True)

  print("input images: ", images.shape)
  print("CNN box - outputs: ", outputs.shape)

  return images, outputs


def weighted_categorical_crossentropy(y_true, y_pred, weights):
  '''
    previous loss function
  '''
  nb_cl = len(weights)
  final_mask = K.zeros_like(y_pred[:, 0])
  y_pred_max = K.max(y_pred, axis=1)
  y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
  y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
  from itertools import product
  for c_p, c_t in product(range(nb_cl), range(nb_cl)):
      final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
  return K.categorical_crossentropy(y_pred, y_true) * final_mask


def loss(y_true, y_pred, weights):
  # scale predictions so that the class probabilities of each sample sum to 1
  y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
  # clipping to remove divide by zero errors
  y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
  # results of loss func
  loss = y_true * K.log(y_pred) * weights
  loss = -K.sum(loss, -1)
  return loss

def build_model(input_shape, weights):
  """Architecture for the xy outputs. Takes a 6-channel image of the environment
    and outputs [x, y, drive/dig] with x,y relative to the active agent's position."""


  downscaleInput = Input(shape=input_shape)
  downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(downscaleInput)
  downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(downscaled)
  downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
  downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = MaxPooling2D(pool_size=(3, 3))(downscaled)
  downscaled = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = MaxPooling2D(pool_size=(3, 3))(downscaled)
  downscaled = Flatten()(downscaled)
  out = Dense(64, activation='sigmoid')(downscaled)
  box = Dense(61, activation='softmax', name='box')(out)
  dig_drive = Dense(1, activation='sigmoid')(out)

  model = Model(inputs=downscaleInput, outputs=[box, dig_drive])
  adam = tf.keras.optimizers.Adam(learning_rate=0.005)
  from functools import partial
  loss1 = partial(loss, weights=weights)
  model.compile(loss=[loss1, 'mse'],  # kullback_leibler_divergence ## categorical_crossentropy
                optimizer=adam,
                metrics=['categorical_accuracy', 'mse'])

  return model


def predict(model=None, data=None, n_examples=5):
  """show inputs together with predictions of CNN,
       either provided as param or loaded from file"""
  print("attempting prediction!")
  if data:
    n_examples = len(data[0])
  if not data:
    images, concat, desired_outputs = load_data()
  else:
    images, concat, desired_outputs = data

  if not model:
    model = tf.keras.models.load_model("saved_models/safetySafe")
  # X1 = images[0][np.newaxis, ...]                        # pretend as if there were multiple input pictures (verbose)
  indeces = random.sample(range(len(images)), n_examples)
  X1 = images[indeces]  # more clever way to write it down
  X2 = concat[indeces]
  desired = desired_outputs[indeces]

  NN_output = model.predict([X1, X2])  # outputs 61x61x2

  # translate the 5 channel input back to displayable images
  orig_img = np.zeros((len(X1), 256, 256))
  for i, image in enumerate(X1):
    print("reconstructing image ", i + 1, "/", n_examples)
    for y, row in enumerate(image):
      for x, cell in enumerate(row):
        for idx, item in enumerate(cell):
          if item == 1:
            orig_img[i][y][x] = idx

  # outputs = np.zeros((len(X1), 256, 256))
  # for img, point in outputs, NN_output:

  # display input images and the 2 waypoint output images (from 2 channels)
  for i in range(len(NN_output)):
    print("agent pos: ", X2[i][-2], X2[i][-1])
    print("desired: ", desired[i])
    print("NN output: ", NN_output[i])
    plt.imshow(orig_img[i])
    plt.title("input image")
    plt.show()


def check_performance(test_data=None, model=None):
  """Check average deviation of x,y,dig/drive outputs from desired
    test outputs, make density plot."""
  if not model:
    model = load("CNNbox")

  images, boxes, dig_drive = test_data
  results = model.predict([images])
  print("results ", results)

  delta_x, delta_y, delta_digdrive = 0, 0, 0
  d_x, d_y, d_digdrive = list(), list(), list()

  for result, desired in zip(boxes, results):
    d_x.append(abs(result[0] - desired[0]))
    d_y.append(abs(result[1] - desired[1]))
    # d_digdrive.append(abs(result[2] - desired[2]))
    delta_x += abs(result[0] - desired[0])
    delta_y += abs(result[1] - desired[1])
    # delta_digdrive += abs(result[2] - desired[2])

  delta_x, delta_y, delta_digdrive = delta_x / len(outputs), delta_y / len(outputs), delta_digdrive / len(outputs)

  print("average Delta X: ", delta_x)
  print("average Delta Y: ", delta_y)
  print("average Delta DD: ", delta_digdrive)

  from scipy.stats import gaussian_kde
  xs = np.linspace(0, 0.2, 200)
  density1 = gaussian_kde(d_x)
  density2 = gaussian_kde(d_y)
  density3 = gaussian_kde(d_digdrive)

  plt.plot(xs, density1(xs))
  plt.plot(xs, density2(xs))
  plt.plot(xs, density3(xs))
  plt.legend(['delta x', 'delta y', 'delta dig/drive'])
  plt.show()


def arrayIndex2WaypointPos(idx):
  size = 5
  timeframe = 20
  x = 0
  cnt = 0
  wp = (0, 0)
  for y in range(-size, size + 1):
    for x in range(-x, x + 1):
      cnt += 1
      if cnt == idx:
        wp = (x, y)
        # print(cnt, "wp", wp)
        if (abs(x) + abs(y)) != 0:
          scale = timeframe / (abs(x) + abs(y))
        else:
          scale = 1
        x = round(scale * x)
        y = round(scale * y)
        wp = (x, y)
    x += 1 if y < 0 else -1

  return wp


# labels_dict : {ind_label: count_label}
# mu : parameter to tune

def create_class_weight(boxID):
  weights_dict = np.asarray(boxID)
  total_val = sum(boxID)
  for i in range(len(boxID)):
      if boxID[i] > 0:
          weights_dict[i] = total_val / boxID[i]
          # weights_dict[i] /= 10
      else:
          weights_dict[i] = total_val
          # weights_dict[i] /= total_val
  # return np.asarray(weights_dict)
  return 1

if __name__ == "__main__":
  # predict()                          # predict with model loaded from file
  # exit()
  architecture_variants = ["xy", "angle", "box"]  # our 3 individual network output variants
  out_variant = architecture_variants[2]
  experiments = ["BASIC", "STOCHASTIC", "WIND", "UNCERTAINTY", "UNCERTAINTY+WIND"]
  experiment = experiments[0]                             # dictates model name

  images, outputs = load_data(out_variant)
  box = []
  dig_drive = []
  boxArr = []
  # images = images[:-9500]
  # outputs = outputs[:-9500]
  for out in outputs:
    box = out[:-1]
    dig_drive.append(out[-1])
    boxArr.append(box)

  boxes = np.asarray(boxArr, dtype=np.float16)
  boxID = [0] * 61
  for box in boxes:
    cnt = 0
    for i in range(len(box)):
      if box[i] == 1:
        boxID[i] += 1

  for p in range(len(boxID)):
    if boxID[p] != 0:
      print(p, "array pos", "waypoint:", arrayIndex2WaypointPos(p), "number of occurances in the dataset:", boxID[p])

  dig_drive = np.asarray(dig_drive, dtype=np.float16)
  print(boxes.shape, dig_drive.shape)
  ## to finish for two things
  # split_point  = int(len(images)*0.2)
  test_data = [images[:20], boxes[:20], dig_drive[:20]]
  images, box, dig_drive = images[20:], boxes[20:], dig_drive[20:]

  # check_performance(test_data)
  # exit()
  ## https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras

  # random labels_dict

  class_weight = create_class_weight(boxID)
  # print("lengths class & boxID", len(class_weight), len(boxID))
  # print("shapes:", images[0].shape)  ##(256, 256, 7)

  model = build_model(images[0].shape, class_weight)
  print(model.summary())
  # exit()

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)


  # class_weight = {0: 0.7,
  #                 1: 0.9, # why y coords less precise??
  #                 2: 0.5}

  history = model.fit(images,  # used to be list of 2 inputs to model
                      [box, dig_drive],
                      batch_size=64,  # 64
                      epochs=40,  # 50
                      shuffle=True,
                      callbacks=[callback],
                      # class_weight=class_weight,
                      validation_split=0.2)  # 0.2


  # print("hist", history.history.keys())
  train_accuracy = history.history['box_categorical_accuracy']
  test_accuracy = history.history['val_box_categorical_accuracy']
  epochs = range(1, len(train_accuracy) + 1)

  plt.plot(epochs, train_accuracy, label='Training Accuracy')
  plt.plot(epochs, test_accuracy, label='Testing Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

  save(model, "CNNbox" + experiment)  # utils
  # check_performance(test_data, model)
  plot_history(history=history)

  # predict(model=model, data=test_data)
