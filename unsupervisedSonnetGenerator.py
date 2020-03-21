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
import HMM

##############################
# Unsupervised HMM
##############################

if __name__ == '__main__':
    allowed_words, syllable_info = parse_syllables('data/Syllable_dictionary.txt')

    n_states_grid = [20]
    n_iter_grid = [50]
    for n_states in n_states_grid:
        for n_iter in n_iter_grid:
            stanzas, word_map, id_map, rhyme_dict = Utility.load_text(allowed_words, 'data/shakespeare.txt')
            print('HMM with {} states, {} iterations, line sequences'.format(n_states, n_iter))
            model = HMM.unsupervised_HMM(stanzas, n_states, n_iter)
            print(model.generate_sonnet(id_map, syllable_info, rhyme_dict))
            print(model.generate_haiku(id_map, syllable_info, True, rhyme_dict))
            print(model.generate_haiku(id_map, syllable_info, False, rhyme_dict))
            wordclouds = states_to_wordclouds(model, word_map)
            anim = animate_emission(model, word_map, M=1)