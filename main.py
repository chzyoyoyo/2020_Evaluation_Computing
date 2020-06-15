import xcs
from xcs.scenarios import ScenarioObserver, Scenario
from xcs.bitstrings import BitString
import logging
import dataset
import numpy as np
import statistics
import copy

class BitcoinProblem(Scenario):
    def __init__(self , input_data):
        self.dataset = input_data
        self.possible_actions = (True, False)
        self.size = input_data.shape[0]
        self.current_index = 0
        self.groundtruth = None
        self.correct = 0

    @property
    def is_dynamic(self):
        return True

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
        if action == self.groundtruth:
            self.correct += 1
        return action == self.groundtruth #reward

if __name__ == '__main__':
    #Get dataset
    my_data = dataset.GetDataset('dataset/ETH_BTC/btc6_15.csv')
    my_data2 = dataset.GetDataset('dataset/ETH_BTC/eth6_15.csv')
    #input_data = dataset.TransformToBinary(my_data , enable_indicator=True , pred_days=1) #for each data, [situation, action]
    #input_data = dataset.TransformToBinary2(my_data , 10 , 1)
    input_data = dataset.TransformToBinary3(my_data, enable_indicator=False, pred_days=1 , comp_days=2)
    # input_data2 = dataset.TransformToBinary3(my_data2, enable_indicator=False, pred_days=1 , comp_days=2)

    input_size = input_data.shape[0]
    traindata = input_data[-1:-int(input_size*0.5):-1]
    testdata = input_data[-int(input_size*0.5)::-1]
    # testdata2 = input_data2[::-1]

    sum_up = 0
    test_number = testdata.shape[0]
    for i in range(test_number):
        if testdata[i][-1] == 1:
            sum_up += 1
    print("Always say True: ", sum_up/test_number)

    #setting algorithm
    algorithm = xcs.XCSAlgorithm()
    algorithm.exploration_probability = 0.1 # exploitation stage
    algorithm.discount_factor = 0.1
    algorithm.crossover_probability = .6
    algorithm.mutation_probability = .05
    algorithm.do_action_set_subsumption = True
    algorithm.idealization_factor = 1 # make the behavior like Q-learning

    result = []
    for i in range(10): #10 indepedent runs
        # -------------------- Training stage ---------------------------
        # setting problem
        scenario = ScenarioObserver(BitcoinProblem(traindata))

        #train model
        model = algorithm.new_model(scenario)
        model.run(scenario , learn=True)

        #print information
        #print(model)
        print('Run',i+1,': training stage correct rate:', scenario.wrapped.correct / scenario.steps)

        #--------------------------- Testing stage -------------------------
        scenario = ScenarioObserver(BitcoinProblem(testdata))
        model.run(scenario , learn=True)
        result.append(scenario.wrapped.correct / scenario.steps)
        print('Run', i+1,': testing stage correct rate:', scenario.wrapped.correct / scenario.steps)

    print('testing stage correct mean:' ,statistics.mean(result), ',variance:' ,  statistics.variance(result))