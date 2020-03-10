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
    def load_text(filename, method):
        '''
        Load the file <filename> into a workable variables

        Arguments:
            filename:   Name of file
            method:     sequence per 0=stanza, 1=line, 2=sonnet

        Returns:
            A:          The transition matrix.
            O:          The observation matrix.
            seqs:       Input sequences.
        '''

        seqs = []
        word_map = {}
        word_counter = 0
        pattern = re.compile(r'[\W_]+', re.UNICODE)

        with open(filename, 'r') as f:
            def parse_stanza(file, num_lines):
                stanza_words = []
                for i in range(num_lines):
                    next_line = file.readline().strip().split(' ')
                    if next_line == '':
                        break
                    stanza_words.extend(next_line)
                for i in range(len(stanza_words)):
                    stanza_words[i] = pattern.sub('', stanza_words[i])
                return stanza_words

            def parse_line(file):
                next_line = file.readline().strip().split(' ')
                for i in range(len(next_line)):
                    next_line[i] = pattern.sub('', next_line[i])
                return next_line

            def parse_sonnet(file):
                poem_words = []
                while True:
                    next_line = file.readline().strip().split(' ')
                    if next_line == '':
                        break
                    poem_words.extend(next_line)
                for i in range(len(poem_words)):
                    poem_words[i] = pattern.sub('', poem_words[i])
                return poem_words

            seq = []
            while True:
                line = f.readline()
                if line == '':
                    break
                if line == '\n':
                    continue

                line_words = line.strip().split(' ')
                if method == 0:
                    if len(line_words) == 1:
                        # New sonnet with 14 lines and length 4/4/4/2 stanzas
                        line_lengths = [4, 4, 4, 2]
                        for length in line_lengths:
                            curr_stanza = parse_stanza(f, length)
                            for word in curr_stanza:
                                if word not in word_map:
                                    word_map[word] = word_counter
                                    word_counter += 1
                                seq.append(word_map[word])

                        # Append stanza sequence to list
                        seqs.append(seq)
                        seq = []
                    else:
                        print('We have a problem with stanza preprocessing')
                elif method == 1:
                    if len(line_words) == 1:
                        continue
                    curr_line = parse_line(f)
                    for word in curr_line:
                        if word not in word_map:
                            word_map[word] = word_counter
                            word_counter += 1
                        seq.append(word_map[word])
                    seqs.append(seq)
                    seq = []
                elif method == 2:
                    if len(line_words) == 1:
                        curr_sonnet = parse_sonnet(f)
                        for word in curr_sonnet:
                            if word not in word_map:
                                word_map[word] = word_counter
                                word_counter += 1
                            seq.append(word_map[word])
                        seqs.append(seq)
                        seq = []
                    else:
                        print('We have a problem with poem preprocessing')

        id_map = {}
        for word in word_map:
            id_map[word_map[word]] = word

        return seqs, word_map, id_map
