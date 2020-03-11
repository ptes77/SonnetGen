from Utility import Utility
from HMM_helper import (
    parse_syllables,
    emission_to_sentence,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission
)
import os
import numpy as np
from IPython.display import HTML
from HMM import unsupervised_HMM
import HMM

##############################
# Unsupervised HMM
##############################

if __name__ == '__main__':
    allowed_words, word_to_syllable = parse_syllables('data/Syllable_dictionary.txt')

    n_states_grid = [24]
    n_iter_grid = [500]
    for n_states in n_states_grid:
        for n_iter in n_iter_grid:
            stanzas, word_map, id_map = Utility.load_text('data/shakespeare.txt', 1, allowed_words)
            print('HMM with {} states, {} iterations, line sequences'.format(n_states, n_iter))
            model = HMM.unsupervised_HMM(stanzas, n_states, n_iter)
            print(model.generate_sonnet(id_map))
            wordclouds = states_to_wordclouds(model, word_map)
            anim = animate_emission(model, word_map, M=1)