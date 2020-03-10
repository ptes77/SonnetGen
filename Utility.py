###
# Utility functions for HMM training
# Miniproject 3
# Based on the Utility.py file from homework set 6
###
import re

class Utility:
    '''
    Utility for the problem files.
    '''

    def __init__(self):
        pass

    @staticmethod
    def load_text(filename):
        '''
        Load the file <filename> into a workable variables

        Arguments:
            filename:   Name of file

        Returns:
            A:          The transition matrix.
            O:          The observation matrix.
            seqs:       Input sequences.
        '''

        stanzas = []
        word_map = {}
        word_counter = 0
        pattern = re.compile(r'[\W_]+', re.UNICODE)

        with open(filename, 'r') as f:
            def parse_stanza(file, num_lines):
                stanza_words = []
                for i in range(num_lines):
                    stanza_words.extend(file.readline().strip().split(' '))
                for i in range(len(stanza_words)):
                    stanza_words[i] = pattern.sub('', stanza_words[i])
                return stanza_words

            stanza_seq = []
            while True:
                line = f.readline()
                if line == '':
                    break
                if line == '\n':
                    continue

                line_words = line.strip().split(' ')
                if len(line_words) == 1:
                    # New sonnet with 14 lines and length 4/4/4/2 stanzas
                    line_lengths = [4, 4, 4, 2]
                    for length in line_lengths:
                        curr_stanza = parse_stanza(f, length)
                        for word in curr_stanza:
                            if word not in word_map:
                                word_map[word] = word_counter
                                word_counter += 1
                            stanza_seq.append(word_map[word])

                    # Append stanza sequence to list
                    stanzas.append(stanza_seq)
                    stanza_seq = []

        id_map = {}
        for word in word_map:
            id_map[word_map[word]] = word

        return stanzas, word_map, id_map
