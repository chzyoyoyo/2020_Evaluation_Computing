import xcs
from xcs.scenarios import ScenarioObserver, Scenario
from xcs.bitstrings import BitString
import logging
import dataset
import numpy as np

class BitcoinProblem(Scenario):
    def __init__(self , input_data):
        self.dataset = input_data
        self.possible_actions = (True, False)
        self.size = input_data.shape[0]
        self.current_index = 0
        self.groundtruth = None

    @property
    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def reset(self):
        self.current_index = 0

    def more(self):
        return self.current_index < self.size

    def sense(self):
        bitstring = BitString(''.join(map(str,self.dataset[self.current_index])))
        situation = bitstring[:-1]
        self.groundtruth = bitstring[-1]
        return situation

    def execute(self, action):
        self.current_index += 1
        return action == self.groundtruth #reward

if __name__ == '__main__':
    #Get dataset
    my_data = dataset.GetDataset('bitcoin.csv')
    input_data = dataset.TransformToBinary(my_data) #for each data, [situation, action]
    np.random.shuffle(input_data)
    # print(input_data.shape())
    input_size = input_data.shape[0]
    traindata = input_data[:int(input_size*0.8)]
    testdata = input_data[int(input_size*0.8):]


    sum_up = 0
    for i in range(input_size):
        if input_data[i][-1] == 1:
            sum_up += 1

    print("sum_up: ", sum_up)
    #-------------------- Training stage ---------------------------
    #setting problem
    scenario = ScenarioObserver(BitcoinProblem(traindata))

    #setting algorithm
    algorithm = xcs.XCSAlgorithm()
    algorithm.exploration_probability = 0.1
    algorithm.ga_threshold = 1
    algorithm.crossover_probability = .5
    algorithm.do_action_set_subsumption = True
    algorithm.do_ga_subsumption = False
    algorithm.wildcard_probability = .998
    algorithm.deletion_threshold = 1
    algorithm.mutation_probability = .002

    #setting logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    #train model
    model = algorithm.new_model(scenario)
    for epoch in range(50): #train 10 epochs
        scenario.reset()
        model.run(scenario , learn=True)

    #print model
    logger.info(model)

    #--------------------------- Testing stage -------------------------
    scenario = ScenarioObserver(BitcoinProblem(testdata))
    model.run(scenario , learn=False)

