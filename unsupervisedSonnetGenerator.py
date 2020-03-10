from Utility import Utility
from HMM_helper import emission_to_sentence
import HMM

##############################
# Unsupervised HMM
##############################

if __name__ == '__main__':
    n_states_grid = range(1, 10)
    stanzas, word_map, id_map = Utility.load_text('data/shakespeare.txt')
    model1 = HMM.unsupervised_HMM(stanzas, 10, 100)
    model2 = HMM.unsupervised_HMM(stanzas, 15, 100)
    print('HMM with 5 states:')
    for i in range(3):
        print(model1.generate_sonnet(id_map))
    print()
    print('HMM with 10 states:')
    for i in range(3):
        print(model2.generate_sonnet(id_map))