import random
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from NNutils import *           # some nice functions for history plotting, saving, loading
from sklearn.model_selection import train_test_split
from global_vars import *

# INPUT
# - we could only pay attention to the non-tree cells to save computation+memory
# - (x,y) positions of agents, active fires, nothing else really necessary?
# - build TWO (agents, active fire) 256x256 matrices from (x,y) positions
# - feed to the network

# OUTPUT
# - (x,y) of the n_agents' waypoints, normalized, must be multiplied with env size + rounded
#   + cut off (!) when outside the env
# - normalize (x,y), but not necessarily universal for all quadratic envs because of fps


def build_model(envsize=256, channels=2, agents=10):
    """Build model using models.Sequential()
    @:param env_size determines the input dimensions
    @:param n_input_matrices amount of 'channels' the CNN can see, for now agents and fires
    @:param n_agents determines how many normalized x,y outputs we will get

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
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())                     # prepare for Dense layer
    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2 * agents))             # x, y for each agent
    model.summary()

    return model

def get_rand_in_out_pair(n_agents = 10, env_size = 256):
    """
    input-output pair is shaped like this:
    pair = (input, output)
    input = (agents, fires)     agents = ((x,y), (x,y), ...)-> length n_agents       fires = ((x,y), (x,y), ...)
    output = waypoints = ((x,y), (x,y), ...)
    x and y integers within env_size
    :return:
    """
    # put n_agents randomly on the training field
    agents = []
    for idx in range(n_agents):
        agents.append(np.array([random.randint(0, env_size), random.randint(0, env_size)]))
    agents = np.array(agents)   # always from lists to np.arrays...

    # same with fires
    n_fires = env_size
    fires = []
    for idx in range(n_fires):
        fires.append(np.array([random.randint(0, env_size), random.randint(0, env_size)]))
    fires = np.array(fires)

    waypoints = []
    for idx in range(n_agents):
        waypoints.append(np.array([random.randint(0, env_size), random.randint(0, env_size)]))
    waypoints = np.array(waypoints)

    # just to make it extra clear (everything in numpy arrays)
    input_data = np.array([agents, fires])  # lists to arrays...
    output_data = waypoints

    return np.array([input_data, output_data])


def load_data(n_training_examples = 100):
    """
    generate some fake data using get_rand_in_out_pair()

    :param n_training_examples: default value 100
    :return: list [inputs, outputs]
    """
    inputs = []
    outputs = []

    for idx in range(n_training_examples):
        pair = get_rand_in_out_pair()
        inputs.append(pair[0])
        outputs.append(pair[1])

    return [np.array(inputs), np.array(outputs)]

def make_input_matrices(inputs_list, env_size=256):
    # make as many input matrices as there is different input information
    inputs = [[np.zeros(shape=(env_size, env_size)) for idx in range(len(inputs_list[0]))] for idx2 in range(len(inputs_list))]
    #           256x256 matrix                                   channels in input image              amount of input examples


    print("shape of inputs: ", np.shape(inputs))
    # loop through all
    for training_example in range(len(inputs_list)):
        for channel in range(len(inputs_list[0])):
            print("ijo")

    return inputs



if __name__ == '__main__':

    model = build_model(env_size, n_channels, n_agents)

    model.compile(optimizer='adam',                 # standard
                  loss='mse',                       # mean squared error
                  metrics=['accuracy']              # accuracy standard, but false positives, FNs etc also possible
    )

    # make zero data
    #inputs = np.zeros(shape=(100, 256, 256, 2))
    #outputs = np.zeros(shape=(100, 20))

    # make random data
    inputs = np.random.randint(0, 2, (100,          # number of input examples
                                      256, 256,     # dims of picture
                                      2)            # channels of picture (agents, fire)
    )
    outputs = np.random.randint(0, 2, (100,
                                       20)          # (x, y) * 10 agents
    )



    # Split+shuffle the data
    #x_train, x_valid, y_train, y_valid = train_test_split(inputs, outputs, test_size=0.33, shuffle=True)

    print(np.shape(inputs))
    print(np.shape(outputs))

    print("input shape of model: ", model.input_shape)
    print("output shape of model: ", model.output_shape)
    history = model.fit(x=inputs,
                        y=outputs,
                        batch_size=16,              # weight adjustments averaged over this amount of samples, powers of 2 standard
                        epochs=5,
                        validation_split=0.2,       # evaluation after training -> for hyper-params
                        shuffle=True                # shuffle before splitting
    )

    plot_history(history=history)







