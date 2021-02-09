# Argument handling
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--run", action="store_true",
                    help="Start the learning process")

parser.add_argument("-tr", "--train_memories", type=int, default=50,
                    help="Number of memories you want to create")

parser.add_argument("-v", "--val_memories", type=int, default=20,
                    help="Number of validation memories you want to create")

parser.add_argument("-te", "--test_episodes", type=int, default=100,
                    help="Number of episodes to test network")

parser.add_argument("-e", "--environment", type=str, default='forest',
                    choices=["forest", "forest_houses", "forest_river", "forest_houses_river"],
                    help="Environment in which you train your agent (containing only forest or also houses and a river)")

parser.add_argument("-s", "--size", type=int, default='10',
                    help="Use the size of the map that you want to use")

# parser.add_argument("-t", "--type", type=str, default="CNN",
#                     choices=["CNN", "CNN_EXTRA", "HI_CNN"],
#                     help="The algorithm to use")

parser.add_argument("-n", "--name", type=str, default="no_name",
                    help="A custom name to give the saved log and model files")

parser.add_argument("-a", "--agents", type=int, default=2,
                    help="Number of agents")

parser.add_argument("-m", "--model", type=str, default='False',
                    help="load a trained model")

parser.add_argument("-vw", "--view", type=str, default='False',
                    help="render a view")

parser.add_argument("-hl", "--history_length", type=int, default=0,
                    help="length of the agents history (enter 10 to select all history")

parser.add_argument("-ex", "--extended", type=str, default='False',
                    help="Include the history of the other agents")

parser.add_argument("-fps", "--fire_propagation_speed", type=int, default=1,
                    help="Should the fire propogation speed be 1 (normal) or 2 (double speed)")

args = parser.parse_args()

if args.fire_propagation_speed not in {1, 2}:
    parser.error("fps should be 1 or 2")

if args.history_length not in {0, 3, 5, 10}:
    parser.error("history length should be 0, 3, 5 or 10 (all history) ")

if args.run and args.name == "no_name":
    parser.error("You should provide a name when running a learning session")

if args.size % 2 == 0:
    parser.error("The size should be an uneven number in order for the data augmentation to work")

if args.test_episodes == 0:
    parser.error("You should specify the number of testing episodes, other wise the model cannot be evaluated")

if args.extended == 'True':
    args.extended = True
else:
    args.extended = False


# Create the simulation
from Simulation.forest_fire import ForestFire

forestfire = ForestFire(args.environment)

# Start learning straight away
if args.run:
    print(f"\nRunning CNN in the "
          f"{args.environment} mode with a size of "
          f"{args.size}*{args.size}.\n")
    # To run the original CNN session
    from CNN import CNN

    Agent = CNN(forestfire, args.name, args.environment, args.size, args.extended, args.history_length)
    counter = 0
    n = 10
    avg = 0
    # test one of the pretrained models
    if args.model == 'True':
        while counter < n:
            forestfire.W.set_board(args.view)
            Agent.test(args.test_episodes, True, args.view)  # test the model n times on 100 new levels
            counter += 1
    # train and test n new models
    else:
        Agent.create_memories(args.train_memories, args.val_memories)  # create data to train and validate
        while counter < n:
            Agent.train()  # train 30 networks
            forestfire.W.set_board(args.view)
            avg += Agent.test(args.test_episodes, False, args.view)  # test 30 models on 100 new levels
            counter += 1
            print("Average accuracy: ", avg / counter)
