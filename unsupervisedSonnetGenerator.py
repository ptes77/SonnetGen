from Utility import Utility
from HMM_helper import emission_to_sentence
import HMM

##############################
# Unsupervised HMM
##############################

if __name__ == '__main__':
    n_states_grid = [12]
    n_iter_grid = [100]
    methods = [0, 1, 2]
    for n_states in n_states_grid:
        for n_iter in n_iter_grid:
            for method in methods:
                stanzas, word_map, id_map = Utility.load_text('data/shakespeare.txt', method)
                print('HMM with {} states, {} iterations, method {}'.format(n_states, n_iter, method))
                model = HMM.unsupervised_HMM(stanzas, n_states, n_iter)
                print(model.generate_sonnet(id_map))
                print()