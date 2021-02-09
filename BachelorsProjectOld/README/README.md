# Wildfire-Control-Python
The custom environment simulates the spread of fire from a 2D, birds-eye-view perspective. 
To contain the fire, the agent (a bulldozer) should learn to dig a road around the fire, enclosing it completely. 
By doing so, we take away the fuel that the fire needs to spread further.\
Follow these instructions with python 3.6.

## Install dependencies:
`pip install -r requirements.txt`

## Create A* files:
`make -C pyastar/`

## Let the algorithm learn and then let it play:
`python main.py -r -tr {amount_of_episodes_to_train} -v {amount_of_episodes_to_validate} -te {amount_of_test_episodes} e-{environment} -t {CNN} -n {name}`

The command above starts the simulation. The terms between brackets{} can be filled in to create your customized
forest fire control system. There are 4 possible environments 'forest', 'forest_river', 'forest_houses' or 
'forest_houses_river'. It will train with the amount of episodes behind -tr, it will evaluate during training based
on the amount of -v episodes. Then finally it will be used to perform in new test episodes (of which no feedback is 
available). From this test episodes you'll receive information about how many times the agent died, isolated the fire
and how much percent of the forest is kept healthy (not burned or digged)

Example, here you play 50 episodes to create training memories, 20 episodes to create validation memories. For the model you've created test it 100 times on random new episodes

`python main.py -r -tr 50 -v 20 -te 100 e- forest_houses -t CNN -n example` 

With the CNN_EXTRA mode you can compare a model trained with the data that comes from the CNN mode plus extra amount of episodes that you specify in this command

`python main.py -r -tr {amount_of_episodes_to_train} -v {amount_of_episodes_to_validate} -te {amount_of_test_runs} e-{environment} -t {CNN_EXTRA} -n {name}`

With the HI_CNN mode you can compare a model with the data that comes from an earlier trained CNN plus extra amount of episodes from fires the model cannot contain already

`python main.py -r -tr {amount_of_episodes_to_train} -v {amount_of_episodes_to_validate} -te {amount_of_test_runs} e-{environment} -t {HI_CNN} -n {name}`


## UPDATED INSTRUCTIONS 2021
The original code has been altered/extended to allow for multiple agents. Some elements of the original code are still in place but serve no purpose anymore. These elements are the different algorithms (CNN_HI and CNN_EXTRA) and the different environments, the simulation now only uses the "forest" environemt. 

The simulation can now also test the influence of including the agents previous positions in the ConvNet input. This is done via the input arguments HL (hisorty length) and EX (extended). HL indicates the amount of previous agent positions that should be inclduded in the input. EX indicates if the input should include the previous positions of all agents or just the "active agent". 

The simulation now also allows for two different fire propagation speeds (FPS). FPS=1 is normal, and FPS=2 is fast. 

The amount of agents can be specified via the argument a.

To see the a trained model in action run the following command:

'python main.py -r -tr 0 -v 0 -te 100 -s 41 -n example -fps 1 -a 2 -m True -vw True'

This runs a simulation which loads a pretrained model with normal FPS, 2 agents and a view to display the simulation. It is then tested on 100 episodes. 

For questions email: nielsrocholl@gmail.com 
