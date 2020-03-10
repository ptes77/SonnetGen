##################################
# Helper file for HMM functions
##################################


def emission_to_sentence(emission, id_map):
    return ' '.join([id_map[word] for word in emission])