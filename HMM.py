########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# # implementation of set 5. Once each part is implemented, you can simply
# # execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# # see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
from numpy.random import choice
from HMM_helper import emission_to_sentence, get_possible_word


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)  # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        probs[0] = self.A_start
        for state in range(self.L):
            probs[1][state] = self.A_start[state] * self.O[state][x[0]]
            seqs[1][state] = str(state)

        for stage in range(2, M + 1):
            # probs[stage][state] = max(prob(y^i | y^{i-1}) * prob(x | state))
            for state in range(self.L):
                poss = []
                for prev in range(self.L):
                    initial = probs[stage - 1][prev]
                    transition = self.A[prev][state]
                    word = self.O[state][x[stage - 1]]
                    prob = initial * transition * word
                    poss.append(prob)
                max_prob = max(poss)
                probs[stage][state] = max_prob
                seqs[stage][state] = seqs[stage - 1][poss.index(max_prob)] + str(state)
        final_prob = probs[-1]
        final_max_prob = max(final_prob)
        max_seq = seqs[-1][final_prob.index(final_max_prob)]
        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)  # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ### TODO: Insert Your Code Here (2Bi)
        ###

        alphas[0] = self.A_start
        for state in range(self.L):
            alphas[1][state] = self.A_start[state] * self.O[state][x[0]]

        for t in range(2, M + 1):
            for state in range(self.L):
                # O_x^i+1,z * sum (j=1^L)alpha_j(i)A_z,j
                alpha_sum = 0
                word = self.O[state][x[t - 1]]
                for prev in range(self.L):
                    alpha = alphas[t - 1][prev]
                    transition = self.A[prev][state]
                    alpha_sum += alpha * transition
                prob = word * alpha_sum
                alphas[t][state] = prob

            if normalize:
                norm = sum(alphas[t])
                for state in range(self.L):
                    alphas[t][state] /= norm

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)  # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ### TODO: Insert Your Code Here (2Bii)
        ###

        betas[M] = [1. for _ in range(self.L)]
        if normalize:
            total = sum(betas[M])
            for i in range(len(betas[M])):
                betas[M][i] /= total

        for t in range(M - 1, 0, -1):
            for state in range(self.L):
                beta_sum = 0
                for prev in range(self.L):
                    beta = betas[t + 1][prev]
                    transition = self.A[state][prev]
                    word = self.O[prev][x[t]]
                    beta_sum += beta * transition * word
                betas[t][state] = beta_sum

            if normalize:
                norm = sum(betas[t])
                for state in range(self.L):
                    betas[t][state] /= norm
        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        ###
        ### TODO: Insert Your Code Here (2C)
        ###
        for a in range(self.L):
            for b in range(self.L):
                a_and_b = 0
                only_a = 0
                for j in range(len(Y)):
                    y = Y[j]
                    for i in range(len(y) - 1):
                        a_and_b += 1 if y[i] == a and y[i + 1] == b else 0
                        only_a += 1 if y[i] == a else 0

                self.A[a][b] = 0 if only_a == 0 else a_and_b / only_a

        # Calculate each element of O using the M-step formulas.

        ###
        ### TODO: Insert Your Code Here (2C)
        ###

        for z in range(self.L):
            for w in range(self.D):
                x_and_y = 0
                only_y = 0
                for i in range(len(X)):
                    x = X[i]
                    y = Y[i]
                    for j in range(len(x)):
                        x_and_y += 1 if y[j] == z and x[j] == w else 0
                        only_y += 1 if y[j] == z else 0
                self.O[z][w] = 0 if only_y == 0 else x_and_y / only_y

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        ###
        ### TODO: Insert Your Code Here (2D)
        ###

        import time
        start = time.time()
        print(N_iters)
        for iters in range(N_iters):
            if iters % 1 == 0:
                print('\r Iteration {}% Complete'.format((100 * iters) // N_iters), end='')

            A_num = [[0. for _ in range(self.L)] for _ in range(self.L)]
            O_num = [[0. for _ in range(self.D)] for _ in range(self.L)]
            A_den = [0. for _ in range(self.L)]
            O_den = [0. for _ in range(self.L)]

            for x in X:
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)
                # print(alphas, betas)
                M = len(x)
                for t in range(1, M + 1):
                    P_curr = [0. for _ in range(self.L)]
                    for state in range(self.L):
                        P_curr[state] = alphas[t][state] * betas[t][state]

                    norm = sum(P_curr)
                    # print(t, M + 1, norm, P_curr)
                    for state in range(len(P_curr)):
                        P_curr[state] /= norm

                    for state in range(self.L):
                        if t != M:
                            A_den[state] += P_curr[state]
                        O_den[state] += P_curr[state]
                        O_num[state][x[t - 1]] += P_curr[state]

                for t in range(1, M):
                    P_curr = [[0. for _ in range(self.L)] for _ in range(self.L)]
                    for j in range(self.L):
                        for i in range(self.L):
                            P_curr[j][i] = alphas[t][j] * self.A[j][i] * self.O[i][x[t]] * betas[t + 1][i]
                    norm = 0
                    for row in P_curr:
                        norm += sum(row)
                    for j in range(self.L):
                        for i in range(self.L):
                            P_curr[j][i] /= norm

                    for j in range(self.L):
                        for i in range(self.L):
                            A_num[j][i] += P_curr[j][i]

            for j in range(self.L):
                for i in range(self.L):
                    self.A[j][i] = A_num[j][i] / A_den[j]

            for j in range(self.L):
                for i in range(self.D):
                    self.O[j][i] = O_num[j][i] / O_den[j]

        print(time.time() - start)

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        ###
        ### TODO: Insert Your Code Here (2F)
        ###

        state = random.choice(range(self.L))
        for t in range(M):
            states.append(state)
            transition = self.A[state]
            observation = self.O[state]
            emission.append(int(choice(range(self.D), 1, p=observation)))
            state = int(choice(range(self.L), 1, p=transition))

        return emission, states


    def generate_line(self, syllable_info, id_map, rhyme_dict, rhyme, num_syllables):
        '''
        Strategy is to generate pairs of rhyming lines and recombine them afterwards
        This method synergizes with our parsing-by-line preprocessing.

        :param syllable_info: dictionary of words to potential syllable count
        :param id_map: dictionary of word id and actual word
        :param rhyme_dict: dictionary of words to words that rhyme
        :param rhyme: boolean that specifies whether to produce two rhyming lines
        :param num_syllables: number of syllables in the line
        :return: array containing words of sentence
        '''
        def generate_single_line(used_syllables):
            syllable_count = used_syllables
            line = []
            state = random.choice(range(self.L))
            # Generate line that takes into account number of syllables already used
            while syllable_count < num_syllables:
                poss_word = None
                transition = self.A[state]
                observation = self.O[state]
                while not poss_word:
                    emission = int(choice(range(self.D), 1, p=observation))
                    poss_word = get_possible_word(emission, syllable_info, id_map, \
                                                  syllable_count, rhyme_dict, num_syllables)

                # Capitalize all standalone 'I's
                if poss_word[0] == 'i':
                    line.append('I')
                else:
                    line.append(poss_word[0])
                syllable_count += poss_word[1]
                state = int(choice(range(self.L), 1, p=transition))
            last_word = line[-1]
            return line, last_word

        used_syllables = 0
        possible_rhymes = []
        # Sometimes the word does not have a corresponding rhyme
        # that is in the allowed words dictionary, so we use
        while len(possible_rhymes) == 0:
            first_line, first_rhyme = generate_single_line(used_syllables)
            if not rhyme:
                return first_line
            possible_rhymes = rhyme_dict[first_rhyme]

        second_rhyme = random.choice(possible_rhymes)
        second_rhyme_syllable_info = syllable_info[second_rhyme]

        # Determine the number of syllables in the second line
        # that is needed for a total of 10 syllables.
        for poss_rhyme in second_rhyme_syllable_info:
            if poss_rhyme[0] == 'E':
                used_syllables = int(poss_rhyme[1])
        if used_syllables == 0:
            used_syllables = int(random.choice(second_rhyme_syllable_info))

        second_line, _ = generate_single_line(used_syllables)
        second_line.append(second_rhyme)

        return first_line, second_line


    def generate_sonnet(self, id_map, syllable_info, rhyme_dict):
        '''
        Generates a sonnet with 14 lines and 10 syllables per line on
        a trained HMM.
        :param id_map: dictionary of word id and actual word
        :param syllable_info: dictionary of words to potential syllable count
        :param rhyme_dict: dictionary of words to words that rhyme
        :return: a string that prints as a 14-line sonnet
        '''
        sonnet = []
        # Generate 7 pairs of rhyming lines
        for i in range(7):
            rhyming_lines = self.generate_line(syllable_info, id_map, rhyme_dict, True, 10)
            for j in range(len(rhyming_lines)):
                rhyming_lines[j][0] = rhyming_lines[j][0].capitalize()
                sonnet.append(' '.join(rhyming_lines[j]))

        rhyming_sonnet = []
        order = [0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 13]
        # Order the rhyming lines so that the sonnet has a rhyme
        # scheme of ababcdcdefefgg.
        for i in range(1, 15):
            rhyming_sonnet.append(sonnet[order[i - 1]])
            # Add punctuation that is similar to that of
            # Shakespearean sonnets
            if i % 4 == 0 or i == 14:
                rhyming_sonnet.append('.\n')
            else:
                rhyming_sonnet.append(',\n')
        return ''.join(rhyming_sonnet)


    def generate_haiku(self, id_map, syllable_info, rhyme, rhyme_dict):
        '''
        Generate a haiku with three lines and 5-7-5 syllable pattern.
        A flag determines whether the haiku follows the aba rhyme scheme.
        :param id_map: dictionary of word id and actual word
        :param syllable_info: dictionary of words to potential syllable count
        :param rhyme: boolean that specifies whether the haiku rhymes
        :param rhyme_dict: dictionary of words to words that rhyme
        :return: a string that prints out a 3-line haiku
        '''

        haiku = []
        if rhyme:
            line1, line3 = self.generate_line(syllable_info, id_map, rhyme_dict, True, 5)
            line2 = self.generate_line(syllable_info, id_map, rhyme_dict, False, 7)
        else:
            line1 = self.generate_line(syllable_info, id_map, rhyme_dict, False, 5)
            line2 = self.generate_line(syllable_info, id_map, rhyme_dict, False, 7)
            line3 = self.generate_line(syllable_info, id_map, rhyme_dict, False, 5)
        lines = [line1, line2, line3]
        for i in range(len(lines)):
            lines[i][0] = lines[i][0].capitalize()
            lines[i] = ' '.join(lines[i])
        haiku = [lines[0], ',\n', lines[1], ',\n', lines[2], '.\n']
        return ''.join(haiku)


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM


def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
