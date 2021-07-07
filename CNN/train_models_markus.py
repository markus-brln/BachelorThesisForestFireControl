import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, \
  Reshape, Activation
from NNutils import *
import tensorflow.keras.backend as K
import sys
# from functools import partial


from getpass import getuser


# tf.random.set_seed(923)
# np.random.seed(923)


def load_data(out_variant, experiment):
  directory = "/home/f118885/data/thesis/" if getuser() == "f118885" else ""
  print("loading data:" + directory + out_variant + experiment + ".npy")
  images = np.load(directory + "images_" + out_variant + experiment + ".npy", allow_pickle=True)
  # concat = np.load("concat_" + out_variant + ".npy", allow_pickle=True)
  outputs = np.load(directory + "outputs_" + out_variant + experiment + ".npy", allow_pickle=True)

  print("input images: ", images.shape)
  print("outputs: ", outputs.shape)

  return images, outputs


def arrayIndex2WaypointPos(idx):
  timeframe = 20
  size = 5
  x = 0
  cnt = 0
  wp = ()
  for y in range(-size, size + 1):
    for x in range(-x, x + 1):
      if cnt == idx:
        ratio = (abs(x) + abs(y)) / size
        timeframe *= ratio
        scale = timeframe / (abs(x) + abs(y))
        newx = (scale * x)
        newy = (scale * y)
        wp = (newx, newy)
      x += 1 if y < 0 else -1
      cnt += 1
  return wp


def create_class_weight(boxID):
  weights_dict = np.asarray(boxID)
  total_val = sum(boxID)
  for i in range(len(boxID)):
    if boxID[i] > 0:
      weights_dict[i] = total_val / boxID[i]
    else:
      weights_dict[i] = total_val
  return np.asarray(weights_dict)


def weighted_loss(weights):
  """Nested loss function to achieve custom loss + class weights.
  https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
  may require special treatment when loading a model, see the link."""

  def my_loss(y_true, y_pred):
    # scale predictions so that the class probabilities of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clipping to remove divide by zero errors
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # results of loss func
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

  return my_loss


def build_model_box(input_shape, weights):
  downscaleInput = Input(shape=input_shape)
  downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(downscaleInput)
  downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(downscaled)
  downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
  downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
  # downscaled = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  # downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
  downscaled = Flatten()(downscaled)
  downscaled = Dropout(0.1)(downscaled)
  out = Dense(16, activation='sigmoid')(downscaled)
  dig_drive = Dense(1, activation='sigmoid', name='dig')(out)
  box = Dense(64, activation='relu')(downscaled)
  box = Dropout(0.1)(box)
  box = Dense(61, activation='softmax', name='box')(box)

  model = Model(inputs=downscaleInput, outputs=[box, dig_drive])
  adam = tf.keras.optimizers.Adam(learning_rate=0.001)  # 0.0005
  # from functools import partial
  # loss1 = partial(loss, weights=weights)
  loss1 = weighted_loss(weights=weights)
  model.compile(loss=[loss1, tf.keras.losses.BinaryCrossentropy()],
                ## categorical_crossentropy  ## tf.keras.losses.BinaryCrossentropy()
                optimizer=adam,
                metrics=['categorical_accuracy', 'binary_crossentropy']
                )
  return model


def build_model_xy(input_shape):
  """Architecture for the xy outputs. Takes a 6-channel image of the environment
  and outputs [x, y, drive/dig] with x,y relative to the active agent's position."""

  downscaleInput = Input(shape=input_shape)
  downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(
    downscaleInput)
  downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
  downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
  downscaled = Flatten()(downscaled)
  out = Dense(48, activation='sigmoid')(downscaled)
  out = Dense(32, activation='sigmoid')(out)
  out = Dense(3)(out)  # nothing specified, so linear output

  model = Model(inputs=downscaleInput, outputs=out)

  adam = tf.keras.optimizers.Adam(learning_rate=0.001)  # initial learning rate faster

  model.compile(loss='mse',
                optimizer=adam,
                metrics=['mse'])

  return model


def build_model_angle(input_shape):
  """Architecture for the xy outputs. Takes a 6-channel image of the environment
      and outputs [x, y, drive/dig] with x,y relative to the active agent's position."""

  downscaleInput = Input(shape=input_shape)
  downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(
    downscaleInput)
  downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
  downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
  downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
  downscaled = Flatten()(downscaled)
  out = Dense(48, activation='sigmoid')(downscaled)
  out = Dense(32, activation='sigmoid')(out)
  out = Dense(4)(out)  # nothing specified, so linear output

  model = Model(inputs=downscaleInput, outputs=out)

  adam = tf.keras.optimizers.Adam(learning_rate=0.001)  # initial learning rate faster

  model.compile(loss='mse',
                optimizer=adam,
                metrics=['mse'])

  return model


def check_performance(test_data, model):
  """ONLY FOR XY, ANGLE VARIANTS
  Check average deviation of x,y,dig/drive outputs from desired
  test outputs, make density plot."""

  images, outputs = test_data
  results = model.predict([images])
  # print("results ", results)

  delta_0, delta_1, delta_2, delta_3 = 0, 0, 0, 0

  for result, desired in zip(outputs, results):
    delta_0 += abs(result[0] - desired[0])
    delta_1 += abs(result[1] - desired[1])
    delta_2 += abs(result[2] - desired[2])
    if len(result) == 4:
      delta_3 += abs(result[3] - desired[3])

  delta_0, delta_1, delta_2, delta_3 = delta_0 / len(outputs), delta_1 / len(outputs), delta_2 / len(
    outputs), delta_3 / len(
    outputs)
  return delta_0, delta_1, delta_2, delta_3


def run_experiments():
  """Choose the architecture variant from the list below, make sure
  you have translated all experiment data files according to your
  architecture:
  - 4 * images_architecture_experiment.npy
  - 4 * outputs_architecture_experiment.npy"""
  import time
  start = time.time()

  n_runs = 30
  architecture_variants = ["xy", "angle", "box"]  # our 3 individual network output variants
  architecture_variant = architecture_variants[2]
  experiments = ["STOCHASTIC", "WINDONLY", "UNCERTAINONLY", "UNCERTAIN+WIND"]
  
  if len(sys.argv) > 1 and int(sys.argv[1]) < len(experiments):
    experiment_nr = int(sys.argv[1])


  for exp, experiment in enumerate(experiments[0+experiment_nr:1+experiment_nr]):

    # performances = open("performance_data/performance" + architecture_variant + experiment + ".txt", mode='w')
    # performances.write("Experiment" + experiment + "\n")
    images, outputs = load_data(architecture_variant, experiment)
    image_shape = images[0].shape
    for run in range(0, n_runs):
      if architecture_variant == 'box':
        print(experiment, "run:", run)
        box = []
        dig_drive = []
        boxArr = []
        for out in outputs:
          box = out[:-1]
          dig_drive.append(out[-1])
          boxArr.append(box)
        boxes = np.asarray(boxArr, dtype=np.float16)
        boxID = [0] * 61
        for box in boxes:
          for i in range(len(box)):
            if box[i] == 1:
              boxID[i] += 1

        class_weight = create_class_weight(boxID)
        dig_drive = np.asarray(dig_drive, dtype=np.float16)

        model = build_model_box(image_shape, class_weight)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_box_categorical_accuracy', patience=15,
                                                    restore_best_weights=True)
        model.fit(images,  # used to be list of 2 inputs to model
                  [boxes, dig_drive],
                  batch_size=64,  # 64
                  epochs=100,  # 50
                  shuffle=True,
                  callbacks=[callback],
                  validation_split=0.2,
                  verbose=2)  # 0.2
      else:
        ##model = build_model_xy(images[0].shape)
        model = build_model_angle(images[0].shape)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(images, outputs,
                  batch_size=64, epochs=100, shuffle=True,
                  callbacks=[callback],
                  validation_split=0.2,
                  verbose=2)
      save(model, "CNN" + architecture_variant + experiment + str(run))
      # performances.write(str(check_performance(test_data, model)) + "\n")

      print(f"model {exp * n_runs + run + 1}/{len(experiments) * n_runs}")
      end = time.time()
      hours, rem = divmod(end - start, 3600)
      minutes, seconds = divmod(rem, 60)
      print("time elapsed:")
      print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

      time_left = ((len(experiments) * n_runs) / (exp * n_runs + run + 1)) * (end - start) - (end - start)

      hours, rem = divmod(time_left, 3600)
      minutes, seconds = divmod(rem, 60)
      print("estimated time left:")
      print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n\n")

    # performances.close()


if __name__ == "__main__":
  run_experiments()
  exit()

# import random
#
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import Input, Model, Sequential
# from tensorflow.keras.layers import concatenate, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv2DTranspose, \
#     Reshape, Activation
# from NNutils import *
# import tensorflow.keras.backend as K
# from functools import partial
#
#
# # tf.random.set_seed(923)
# # np.random.seed(923)
#
#
# def load_data(out_variant, experiment):
#     print("loading data:" + out_variant + experiment + ".npy")
#     images = np.load("images_" + out_variant + experiment + ".npy", allow_pickle=True)
#     # concat = np.load("concat_" + out_variant + ".npy", allow_pickle=True)
#     outputs = np.load("outputs_" + out_variant + experiment + ".npy", allow_pickle=True)
#
#     print("input images: ", images.shape)
#     print("outputs: ", outputs.shape)
#
#     return images, outputs
#
# def arrayIndex2WaypointPos(idx):
#   timeframe = 20
#   size = 5
#   x = 0
#   cnt = 0
#   wp = ()
#   for y in range(-size, size + 1):
#     for x in range(-x, x + 1):
#       if cnt == idx:
#         ratio = (abs(x) + abs(y)) / size
#         timeframe *= ratio
#         scale = timeframe / (abs(x) + abs(y))
#         newx = (scale * x)
#         newy = (scale * y)
#         wp = (newx, newy)
#       x += 1 if y < 0 else -1
#       cnt += 1
#   return wp
#
# def create_class_weight(boxID):
#   weights_dict = np.asarray(boxID)
#   total_val = sum(boxID)
#   for i in range(len(boxID)):
#       if boxID[i] > 0:
#           weights_dict[i] = total_val / boxID[i]
#       else:
#           weights_dict[i] = total_val
#   return np.asarray(weights_dict)
#
#
#
# def loss(y_true, y_pred, weights):
#   # scale predictions so that the class probabilities of each sample sum to 1
#   y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#   # clipping to remove divide by zero errors
#   y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#   # results of loss func
#   loss = y_true * K.log(y_pred) * weights
#   loss = -K.sum(loss, -1)
#   return loss
#
# def build_model_box(input_shape, weights):
#     downscaleInput = Input(shape=input_shape)
#     downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(downscaleInput)
#     downscaled = Conv2D(filters=16, kernel_size=(2, 2),  strides=(1, 1), activation="relu", padding="same")(downscaled)
#     downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
#     downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
#     downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
#     downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
#     downscaled = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
#     downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
#     downscaled = Flatten()(downscaled)
#     downscaled = Dropout(0.03)(downscaled)
#     out = Dense(8, activation='sigmoid')(downscaled)
#     dig_drive = Dense(1, activation='sigmoid', name='dig')(out)
#     box = Dense(64, activation='relu')(downscaled)
#     box = Dense(61, activation='softmax', name='box')(box)
#
#     model = Model(inputs=downscaleInput, outputs=[box, dig_drive])
#     adam = tf.keras.optimizers.Adam(learning_rate=0.001)#0.0005
#     loss1 = partial(loss, weights=weights)
#     model.compile(loss=[loss1, tf.keras.losses.BinaryCrossentropy()], ## categorical_crossentropy  ## tf.keras.losses.BinaryCrossentropy()
#                   optimizer=adam,
#                   metrics=['categorical_accuracy', 'binary_crossentropy']
#                   )
#     return model
#
#
# def build_model_xy(input_shape):
#     """Architecture for the xy outputs. Takes a 6-channel image of the environment
#     and outputs [x, y, drive/dig] with x,y relative to the active agent's position."""
#
#     downscaleInput = Input(shape=input_shape)
#     downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(
#         downscaleInput)
#     downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
#     downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
#     downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
#     downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
#     downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
#     downscaled = Flatten()(downscaled)
#     out = Dense(48, activation='sigmoid')(downscaled)
#     out = Dense(32, activation='sigmoid')(out)
#     out = Dense(3)(out)  # nothing specified, so linear output
#
#     model = Model(inputs=downscaleInput, outputs=out)
#
#     adam = tf.keras.optimizers.Adam(learning_rate=0.001)  # initial learning rate faster
#
#     model.compile(loss='mse',
#                   optimizer=adam,
#                   metrics=['mse'])
#
#     return model
#
#
# def build_model_angle(input_shape):
#     """Architecture for the xy outputs. Takes a 6-channel image of the environment
#         and outputs [x, y, drive/dig] with x,y relative to the active agent's position."""
#
#     downscaleInput = Input(shape=input_shape)
#     downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation="relu", padding="same")(
#         downscaleInput)
#     downscaled = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
#     downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
#     downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
#     downscaled = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="relu", padding="same")(downscaled)
#     downscaled = MaxPooling2D(pool_size=(2, 2))(downscaled)
#     downscaled = Flatten()(downscaled)
#     out = Dense(48, activation='sigmoid')(downscaled)
#     out = Dense(32, activation='sigmoid')(out)
#     out = Dense(4)(out)  # nothing specified, so linear output
#
#     model = Model(inputs=downscaleInput, outputs=out)
#
#     adam = tf.keras.optimizers.Adam(learning_rate=0.001)  # initial learning rate faster
#
#     model.compile(loss='mse',
#                   optimizer=adam,
#                   metrics=['mse'])
#
#     return model
#
#
# def check_performance(test_data, model):
#     """Check average deviation of x,y,dig/drive outputs from desired
#     test outputs, make density plot."""
#
#     images, outputs = test_data
#     results = model.predict([images])
#     # print("results ", results)
#
#     delta_0, delta_1, delta_2, delta_3 = 0, 0, 0, 0
#
#     for result, desired in zip(outputs, results):
#         delta_0 += abs(result[0] - desired[0])
#         delta_1 += abs(result[1] - desired[1])
#         delta_2 += abs(result[2] - desired[2])
#         if len(result) == 4:
#             delta_3 += abs(result[3] - desired[3])
#
#     delta_0, delta_1, delta_2, delta_3= delta_0 / len(outputs), delta_1 / len(outputs), delta_2 / len(outputs), delta_3 / len(
#         outputs)
#     return delta_0, delta_1, delta_2, delta_3
#
#
# def run_experiments():
#     """Choose the architecture variant from the list below, make sure
#     you have translated all experiment data files according to your
#     architecture:
#     - 4 * images_architecture_experiment.npy
#     - 4 * outputs_architecture_experiment.npy"""
#     import time
#     start = time.time()
#
#     n_runs = 12
#     architecture_variants = ["xy", "angle", "box"]  # our 3 individual network output variants
#     architecture_variant = architecture_variants[2]
#     experiments = ["STOCHASTIC", "WINDONLY", "UNCERTAINONLY", "UNCERTAIN+WIND"]
#
#     for exp, experiment in enumerate(experiments):
#
#         performances = open("performance_data/performance" + architecture_variant + experiment + ".txt", mode='w')
#         performances.write("Experiment" + experiment + "\n")
#
#         images, outputs = load_data(architecture_variant, experiment)
#
#         for run in range(0, n_runs):
#           if architecture_variant == 'box':
#             images, outputs = load_data(architecture_variant, experiment)
#             box = []
#             dig_drive = []
#             boxArr = []
#             for out in outputs:
#               box = out[:-1]
#               dig_drive.append(out[-1])
#               boxArr.append(box)
#             boxes = np.asarray(boxArr, dtype=np.float16)
#             boxID = [0] * 61
#             for box in boxes:
#               for i in range(len(box)):
#                 if box[i] == 1:
#                   boxID[i] += 1
#               class_weight = create_class_weight(boxID)
#               dig_drive = np.asarray(dig_drive, dtype=np.float16)
#               print(experiment, "run:", run)
#               test_data = [images[:100], boxes[:100], dig_drive[:100]]
#               images, box, dig_drive = images[100:], boxes[100:], dig_drive[100:]
#               model = build_model_box(images[0].shape, class_weight)
#               callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#               model.fit(images,  # used to be list of 2 inputs to model
#                         [box, dig_drive],
#                         batch_size=64,  # 64
#                         epochs=100,  # 50
#                         shuffle=True,
#                         callbacks=[callback],
#                         validation_split=0.2,
#                         verbose=2)  # 0.2
#             else:
#               ##model = build_model_xy(images[0].shape)
#               model = build_model_angle(images[0].shape)
#
#               callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#               model.fit(images, outputs,
#                         batch_size=64, epochs=100, shuffle=True,
#                         callbacks=[callback],
#                         validation_split=0.2,
#                         verbose=2)
#
#             save(model, "CNN" + architecture_variant + experiment + str(run))
#             performances.write(str(check_performance(test_data, model)) + "\n")
#
#             print(f"model {exp * n_runs + run + 1}/{len(experiments) * n_runs}")
#             end = time.time()
#             hours, rem = divmod(end - start, 3600)
#             minutes, seconds = divmod(rem, 60)
#             print("time elapsed:")
#             print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
#
#             time_left = ((len(experiments) * n_runs) / (exp * n_runs + run + 1)) * (end - start) - (end - start)
#
#             hours, rem = divmod(time_left, 3600)
#             minutes, seconds = divmod(rem, 60)
#             print("estimated time left:")
#             print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n\n")
#
#         performances.close()
#
#
# if __name__ == "__main__":
#     run_experiments()
#     exit()
