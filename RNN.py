import keras, re
import keras.utils
import numpy as np
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import LSTM
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import LambdaCallback, EarlyStopping


def get_sequences(filename, input_length, skip):
    text = open("data/Shakespeare.txt").read().lower()
    text = re.sub('\d+', '', text)  # Get rid of line numbers
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('\n\n\n', '')
    text = text.replace('  ', ' ')
    text = text.replace('-', ' ')

    seqs = []
    for i in range(input_length, len(text), skip):
        curr_seq = text[i - input_length:i + 1]
        seqs.append(curr_seq)
    return seqs

seqs = get_sequences("data/shakespeare.txt", 40, 1)
new_seqs = np.array(seqs)
chars = sorted(list(set(' '.join(seqs))))
maps = dict((char, idx) for idx,char in enumerate(chars))
inverse_map = dict((idx, char) for idx,char in enumerate(chars)) # to get back after text generation
number_sequences = list()
for line in new_seqs:
    curr_seq = [maps[char] for char in line]
    number_sequences.append(curr_seq)
encoded_seqs = np.array(number_sequences)

orig_X,orig_y = encoded_seqs[:,:-1],encoded_seqs[:,-1]
# one hot encode y values
y = keras.utils.to_categorical(orig_y, num_classes = len(maps))
# #normalize X and prepare for keras format
# X = np.reshape(orig_X, (len(orig_X), len(orig_X[0]), 1))
# X = X/float(len(maps))
X = np.array([keras.utils.to_categorical(x, num_classes=len(maps)) for x in orig_X])

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, y, epochs=200, batch_size=64)

def print_random():
    # pick a random seed
    start = np.random.randint(0, len(X) - 1)
    base = list(orig_X[start])
    print("Seed:")
    print("\"", ''.join([inverse_map[value] for value in base]), "\"")

    # generate characters
    for i in range(500):
        # print(base)
        # print([''.join(inverse_map[value] for value in base)])
        temp = keras.utils.to_categorical(base, num_classes=len(maps))
        temp = np.reshape(temp, (1, temp.shape[0], temp.shape[1]))
        prediction = model.predict(temp, verbose=0)
        idx = np.argmax(prediction[0])
        sys.stdout.write(inverse_map[idx])
        base.append(idx)
        base = base[1:len(base)]
    print("\nDone.")

def sample(preds, temp=1.0):
    preds = np.log(preds) / temp
    preds = np.exp(preds) / np.sum(np.exp(preds))
    preds -= 0.000001
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

def print_fixed_seed():
    for temperature in [1.5, 0.75, 0.25]:
        base = "shall i compare thee to a summer's day?\n"
        print("TEMPERATURE: ", temperature)
        for i in range(650):
            x_pred = np.zeros((1, 40, len(chars)))
            for t, char in enumerate(base[-40:]):
                x_pred[0, t, maps[char]] = 1.
            preds = model.predict(x_pred)[0]
            idx = sample(preds, temperature)
            base += inverse_map[idx]
        print(base[40:])