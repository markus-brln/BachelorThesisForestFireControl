import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
# from keras.engine.saving import model_from_json
from tensorflow.keras import layers, models

def plot_np_image(image):
  channels = np.dsplit(image.astype(dtype=np.float32), len(image[0][0]))
  f, axarr = plt.subplots(2, 4)
  axarr[0, 0].imshow(np.reshape(channels[0], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[0, 0].set_title("active fire")
  axarr[0, 1].imshow(np.reshape(channels[1], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[0, 1].set_title("fire breaks")
  axarr[0, 2].imshow(np.reshape(channels[2], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[0, 2].set_title("wind dir (uniform)")
  axarr[1, 0].imshow(np.reshape(channels[3], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[1, 0].set_title("wind speed (uniform)")
  axarr[1, 1].imshow(np.reshape(channels[4], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[1, 1].set_title("other agents")
  axarr[1, 2].imshow(np.reshape(channels[5], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[1, 2].set_title("active agent x")
  axarr[1, 3].imshow(np.reshape(channels[6], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[1, 3].set_title("active agent y")
  print("x, y pos of active agent: ", channels[0][0][5], channels[0][0][6])
  plt.show()

def plot_data(data):
  for dat in data:
    print("wind dir:", dat[1])
    print("wind speed:", dat[2])
    print("agent info:",dat[3])
    plt.imshow(dat[0])
    plt.show()
  exit()

def unison_shuffled_copies(a, b):
    # from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def n_samples(list_of_lists, k):
    """Takes k random samples from all lists inside the given list."""
    if k > len(list_of_lists[0]):
        print("list not long enough! returning full list.")
        return list_of_lists

    indeces = np.random.permutation(k)
    n_lists = len(list_of_lists)
    extracted =[]# [list() for i in range(n_lists)]
    reduced = []#[list() for i in range(n_lists)]

    shuffle_seed = random.random()
    for single_list in list_of_lists:
        random.shuffle(single_list, shuffle_seed)

    """for i in range(len(list_of_lists[0])):
        for j in range(n_lists):
            print(j, i)
            if i in indeces:
                extracted[j].append(list_of_lists[i][j].pop(i))"""


    return extracted, reduced



def plot_history(history):
    # function taken from https://github.com/musikalkemist/DeepLearningForAudioWithPython
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    #axs[0].plot(history.history["accuracy"], label="train accuracy")
    #axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    #axs[0].set_ylabel("Accuracy")
    #axs[0].legend(loc="lower right")
    #axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def plot_history_box(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model

        Pro Tipp:     print(history.history.keys())
                      if you want to find out what you can actually plot :)

        inspiration taken from https://github.com/musikalkemist/DeepLearningForAudioWithPython
    """
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["box_categorical_accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_box_categorical_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("box_categorical_accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("val_box_categorical_accuracy")

    # create error sublpot
    axs[1].plot(history.history["box_loss"], label="train error")
    axs[1].plot(history.history["val_box_loss"], label="validation error")
    axs[1].set_ylabel("box_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("val_box_loss")

    plt.show()


# https://github.com/musikalkemist/DeepLearningForAudioWithPython
# https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf
def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add dimensions to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ..., np.newaxis]

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    #predicted_index = np.argmax(prediction, axis=1)
    plt.imshow(X)
    plt.show()

    plt.imshow(prediction)
    plt.show()


#### Saving a Network
def save(model, filename):
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    model_json = model.to_json()
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    with open('saved_models' + os.sep + filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('saved_models' + os.sep + filename + ".h5")
    print("saving model " + filename)
    #model.save('saved_models\\' + filename)


#### Loading a Network
def load(filename):
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    print("loading model " + filename)
    # load json and create model
    json_file = open('saved_models' + os.sep + filename + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(model_json)
    # load weights into new model
    model.load_weights('saved_models' + os.sep + filename + ".h5")
    print("Loaded model from disk")

    #json_model_file = open(os.path.join(self.model_path, name + '.json'), "r").read()
    #model = model_from_json(open('saved_models/' + filename).read())
    #model.load_weights(os.path.join(os.path.dirname('saved_models/' + filename), 'model_weights.h5'))

    #model = tf.keras.models.load_model('saved_models/' + filename)
    return model


def save_history(file_name, history):
    np.save(file_name, history.history)


def load_history(filename):
    return np.load(filename, allow_pickle=True).item()

def predict(model=None, data=None, n_examples=5):
    """NOT USED show inputs together with predictions of CNN,
       either provided as param or loaded from file"""
    if data:
        n_examples = len(data[0])
    if not data:
        images, concat, desired_outputs = load_data()
    else:
        images, concat, desired_outputs = data

    if not model:
        model = tf.keras.models.load_model("saved_models/safetySafe")
    #X1 = images[0][np.newaxis, ...]                        # pretend as if there were multiple input pictures (verbose)
    indeces = random.sample(range(len(images)), n_examples)
    X1 = images[indeces]                                        # more clever way to write it down
    X2 = concat[indeces]
    desired = desired_outputs[indeces]

    NN_output = model.predict([X1, X2])                        # outputs 61x61x2

    # translate the 5 channel input back to displayable images
    orig_img = np.zeros((len(X1), 256, 256))
    for i, image in enumerate(X1):
        print("reconstructing image ", i+1, "/", n_examples)
        for y, row in enumerate(image):
            for x, cell in enumerate(row):
                for idx, item in enumerate(cell):
                    if item == 1:
                        orig_img[i][y][x] = idx

    #outputs = np.zeros((len(X1), 256, 256))
    #for img, point in outputs, NN_output:


    # display input images and the 2 waypoint output images (from 2 channels)
    for i in range(len(NN_output)):
        print("agent pos: ", X2[i][-2], X2[i][-1])
        print("desired: ", desired[i])
        print("NN output: ", NN_output[i])
        plt.imshow(orig_img[i])
        plt.title("input image")
        plt.show()
