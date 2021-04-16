from NNutils import *           # some nice functions for history plotting, saving, loading
                                # building model
from global_vars import *       # like env_size, n_channels, n_agents

# INPUT
# - we could only pay attention to the non-tree cells to save computation+memory
# - (x,y) positions of agents, active fires, nothing else really necessary?
# - build TWO (agents, active fire) 256x256 matrices from (x,y) positions
# - feed to the network

# OUTPUT
# - (x,y) of the n_agents' waypoints, normalized, must be multiplied with env size + rounded
#   + cut off (!) when outside the env
# - normalize (x,y), but not necessarily universal for all quadratic envs because of fps


if __name__ == '__main__':
    model = build_model(env_size, n_channels, n_agents) #

    model.compile(optimizer='adam',                 # standard
                  loss='mse',                       # mean squared error
                  metrics=['accuracy']              # accuracy standard, but false positives, FNs etc also possible
    )

    # make zero data
    #inputs = np.zeros(shape=(100, 256, 256, 2))
    #outputs = np.zeros(shape=(100, 20))

    # make random data
    inputs = np.random.randint(0, 2, (1000,          # number of input examples
                                      256, 256,     # dims of picture
                                      2)            # channels of picture (agents, fire)
    )
    outputs = np.random.randint(0, 2, (1000,
                                       20)          # (x, y) * 10 agents
    )

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







