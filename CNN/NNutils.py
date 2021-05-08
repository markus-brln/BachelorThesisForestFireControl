import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

def unison_shuffled_copies(a, b):
    # from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def build_model(envsize=256, channels=2, agents=10):
    """Build model using models.Sequential()
    @:param envsize determines the input dimensions
    @:param channels amount of 'channels' the CNN can see, for now agents and fires
    @:param agents determines how many normalized x,y outputs we will get

    @:return the model that has to be compile()d, fit(), test()ed
    """
    model = models.Sequential()
    # to make clear what numbers represent what:
    model.add(layers.Conv2D(filters=32,             # convolution channels in output of this layer
                            # https://stackoverflow.com/questions/51180234/keras-conv2d-filters-vs-kernel-size
                            kernel_size=(3, 3),     # filter that walks in 'strides' over input images
                            activation='relu',      # max(0, input), so quite fast
                            input_shape=(envsize, envsize, channels))
    )
    # the layers in between are subject to experimentation
    model.add(layers.MaxPooling2D((5,5)))           # reduce connections
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())                     # prepare for Dense layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2 * agents))             # x, y for each agent
    model.summary()

    return model


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
    print("saving model " + filename)
    model.save('saved_models\\' + filename)


#### Loading a Network
def load(filename):
    print("loading model " + filename)
    model = tf.keras.models.load_model('saved_models/' + filename)
    return model


def save_history(file_name, history):
    np.save(file_name, history.history)


def load_history(filename):
    return np.load(filename, allow_pickle=True).item()

