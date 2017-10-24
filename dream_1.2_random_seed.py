

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import pretty_midi
from data import midi_numberer, pno_numberer, grade_and_chorder, pno_pitcher
from algs import midi_generator

mari_287_to_310 = [['d4'],['g4','c5'],['f4'],['af3','ef5'],['d4'],['g4','c5'],['f4'],['af3','ef5'],['f4'],['g5','ef5'],\
                   ['df4'],['c6','ef5'],['f4'],['bf3','ef4'],['d4'],['gs5','a5','ef5'],['d4'],['b3','c5'],['ef4'],['c6','df5'],\
                   ['f5'],['bf4','ef6'],['df5'],['d4','c6'],['ef4'],['af4','df5'],['fs4'],['a3','e5'],['ef4'],['af4','df5'],\
                   ['fs4'],['a3','e5'],['fs4'],['af5','e5'],['d4'],['cs6','e5'],['fs4'],['b3','e5'],['ds4'],['a5','bf5','e5'],['ds4'],['c4','cs5'],\
                   ['e4'],['cs6','d5'],['fs5'],['b4','e6'],['d5'],['ef4','cs6']]
simple_pno_mari = []
for i in mari_287_to_310:
    partial=[]
    for j in i:
        partial.append(pno_numberer(j))
    simple_pno_mari.append(partial)
simple_keyboard_mari= np.zeros((len(mari_287_to_310),88),dtype=bool)
for i,item in enumerate(simple_pno_mari):
    for j in item:
        simple_keyboard_mari[i][j] = 1.0

print('score length:', len(simple_keyboard_mari))

# cut the text in semi-redundant sequences of maxlen characters


maxlen = 12
step = 1
phrases = []
next_notes = []

for i in range(0,len(simple_keyboard_mari) - maxlen, step):
    phrases.append(simple_keyboard_mari[i: i + maxlen])
    next_notes.append(simple_keyboard_mari[i + maxlen])

print('number of phrases:', len(phrases))

print('Vectorization...')

X = [phrases[i].tolist() for i in range(len(phrases))]
y = [next_notes[i].tolist() for i in range(len(next_notes))]



# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, 88)))
model.add(Dense(88))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=200)

    start_index = random.randint(0, len(simple_keyboard_mari) - maxlen - 1)

    for diversity in [1.0]:
        print()

        generated = ''
        randoms=[random.choice([i for i in range(len(mari_287_to_310))]) for j in range(maxlen)]
        phrase = str([mari_287_to_310[i] for i in randoms])
        p=np.array([[simple_keyboard_mari[i] for i in randoms]])
        generated += phrase
        print('----- Generating with seed: "' + phrase + '"')
        sys.stdout.write(generated)
        nexties = []

        for i in range(100):
            x = p

            preds = model.predict(x, verbose=0)[0]
            next=[]
            rounded_preds = np.zeros(88, bool)
            for i,item in enumerate(preds):
                if item > 0.1:
                    next.append(i)
                    rounded_preds[i] = 1

            p=np.array([np.append(p[0],np.array([rounded_preds]),axis=0)])
            p=np.delete(p,0,1)

            nexties.append([pno_pitcher(i) for i in next])

        print()
        # print(nexties)


tot_music = [mari_287_to_310[i] for i in randoms] + nexties

notes=[]
time=0.0
for i in tot_music:
    if len(i) == 1:
        dur = 0.2
        for j in i:
            notes.append([pno_numberer(j)*2,time,dur,20])
        time += dur
    else:
        dur = 3.8
        for j in i:
            notes.append([pno_numberer(j)*2,time,dur,30])
        time += dur

midi_generator(notes, 'feldman_dream_1_random_seed.mid', sustain='yes')

# midi_generator(notes,'feldman_dream_1.mid',sustain = 'no')
