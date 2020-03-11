###
# Utility functions for HMM training
# Miniproject 3
# Based on the Utility.py file from homework set 6
###
import re


def parse_stanza(file, num_lines, pattern):
    stanza_words = []
    for i in range(num_lines):
        next_line = file.readline().strip().lower()
        if next_line == '':
            break
        next_line = next_line.split(' ')
        stanza_words.extend(next_line)
    for i in range(len(stanza_words)):
        stanza_words[i] = pattern.sub('', stanza_words[i])
    return stanza_words


def parse_line(file, pattern):
    line_words = []
    while True:
        next_line = file.readline().strip().lower()
        if next_line == '':
            break
        next_line = next_line.split(' ')
        for i in range(len(next_line)):
            next_line[i] = pattern.sub('', next_line[i])
        line_words.append(next_line)
    return line_words


def parse_sonnet(file, pattern):
    sonnet_words = []
    while True:
        next_line = file.readline().strip().lower()
        if next_line == '':
            break
        next_line = next_line.split(' ')
        sonnet_words.extend(next_line)
    for i in range(len(sonnet_words)):
        sonnet_words[i] = pattern.sub('', sonnet_words[i])
    return sonnet_words


class Utility:
    '''
    Utility for the problem files.
    '''

    def __init__(self):
        pass

    @staticmethod
    def load_text(filename, method, allowed_words):
        '''
        Load the file <filename> into a workable variables

        Arguments:
            filename:   Name of file
            method:     sequence per 0=stanza, 1=line, 2=sonnet
            allowed_words:  List of allowed words

        Returns:
            A:          The transition matrix.
            O:          The observation matrix.
            seqs:       Input sequences.
        '''

        seqs = []
        word_map = {}
        word_counter = 0
        pattern = re.compile(r"[^\w']+", re.UNICODE)

        with open(filename, 'r') as f:
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
                            curr_stanza = parse_stanza(f, length, pattern)
                            for word in curr_stanza:
                                if word in allowed_words:
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
                        curr_line = parse_line(f, pattern)
                        for single_line in curr_line:
                            for word in single_line:
                                if word in allowed_words:
                                    if word not in word_map:
                                        word_map[word] = word_counter
                                        word_counter += 1
                                    seq.append(word_map[word])
                            seqs.append(seq)
                            seq = []
                elif method == 2:
                    if len(line_words) == 1:
                        curr_sonnet = parse_sonnet(f, pattern)
                        for word in curr_sonnet:
                            if word in allowed_words:
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