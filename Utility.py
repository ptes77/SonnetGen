###
# Utility functions for HMM training
# Miniproject 3
# Based on the Utility.py file from homework set 6
###
import re
from collections import defaultdict


def parse_line(file, pattern):
    line_words = []
    while True:
        next_line = file.readline().strip().lower()

        # Keeps parsing lines until we reach an empty line.
        if next_line == '':
            break
        next_line = next_line.split(' ')

        # For each line, we take out all punctuation so that the
        # word corpus contains only words.
        for i in range(len(next_line)):
            next_line[i] = pattern.sub('', next_line[i])
        line_words.append(next_line)
    return line_words


class Utility:
    '''
    Utility for the problem files.
    '''

    def __init__(self):
        pass

    @staticmethod
    def load_text(allowed_words, filename, *filenames):
        '''
        :param allowed_words: List of allowed words
        :param filename: Name of file
        :param filenames: Name(s) of extra files
        :return:
            A:          The transition matrix.
            O:          The observation matrix.
            seqs:       Input sequences.
        '''

        # Initialize variables and parsing regex command
        seqs = []
        word_map = {}
        rhyme_dict = defaultdict(lambda: [])
        word_counter = 0
        pattern = re.compile(r"[^\w']+", re.UNICODE)

        def last_word_of_line(lines, line_num):
            return lines[line_num][-1]

        files = [filename]
        for file in filenames:
            files.append(file)
        print(files)

        for name in files:
            with open(name, 'r') as f:
                seq = []
                while True:
                    line = f.readline()
                    # End of document
                    if line == '':
                        break
                    # Lines between sonnets
                    if line == '\n':
                        continue

                    line_words = line.strip().split(' ')
                    # Parse the sonnet if at the beginning of the sonnet,
                    # which is represented by the sonnet number
                    if len(line_words) == 1:
                        sonnet_lines = parse_line(f, pattern)
                        num_lines = len(sonnet_lines)
                        # If the number of lines is not 14, then we cannot
                        # easily find the rhyming words, so we throw out
                        # this data point.
                        if num_lines != 14:
                            continue

                        # rhyme scheme: ababcdcdefefgg
                        last_words = [last_word_of_line(sonnet_lines, i) for i in range(num_lines)]
                        pairs = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]
                        for rhyme in pairs:
                            first, second = last_words[rhyme[0]], last_words[rhyme[1]]
                            # Only add rhyme to rhyme dictionary if both itself and
                            # its pair rhyme are in the syllable dictionary.
                            if first in allowed_words and second in allowed_words:
                                rhyme_dict[first].append(second)
                                rhyme_dict[second].append(first)

                        # Add words to the word sequence if they are allowed
                        for single_line in sonnet_lines:
                            for word in single_line:
                                if word in allowed_words:
                                    if word not in word_map:
                                        word_map[word] = word_counter
                                        word_counter += 1
                                    seq.append(word_map[word])
                            seqs.append(seq)
                            seq = []

        id_map = {}
        for word in word_map:
            id_map[word_map[word]] = word

        return seqs, word_map, id_map, rhyme_dict
