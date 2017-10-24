# with chord info as first items in neural net [1,0,0,0,0,0] means chord size of 0, [0,1,0,0,0,0] means chord size of 1, etc.

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

chord_size=[len(i) for i in simple_pno_mari]
chord_size_nn=np.zeros([len(chord_size),8],bool)
for i,item in enumerate(chord_size):
    chord_size_nn[i][item] = 1

simple_keyboard_mari=np.concatenate((chord_size_nn,simple_keyboard_mari),axis=1)



print('score length:', len(simple_keyboard_mari))

# cut the text in semi-redundant sequences of maxlen characters


maxlen = 24
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
model.add(LSTM(128, input_shape=(maxlen, 88 + 8)))
model.add(Dense(88 + 8))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)



# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=300)

    start_index = random.randint(0, len(simple_keyboard_mari) - maxlen - 1)

    for diversity in [1.0]:
        print()

        generated = ''
        really_random = []
        for i in range(maxlen):
            rr = np.zeros(96, bool)
            for j in range(96):
                t = random.choice([0, 1])
                if t == 1:
                    rr[j] = 1
            really_random.append(rr)
        p = np.array([really_random])

        phrase = 'random shit'
        generated += phrase
        print('----- Generating with seed: "' + phrase + '"')
        sys.stdout.write(generated)
        nexties = []

        for i in range(100):


            preds = model.predict(p, verbose=0)[0]
            chord_size_preds=preds[:8]
            #chord_size = np.nonzero(chord_size_preds == max(chord_size_preds))[0][0]
            chord_size = np.random.choice(len(chord_size_preds),1,p=[i / sum(chord_size_preds) for i in chord_size_preds])[0]
            chord_size_zeroed = np.zeros(8,bool)
            chord_size_zeroed[chord_size] = 1
            preds = preds[8:]
            next=[]
            rounded_preds = np.zeros(88, bool)
            sorted_preds = np.sort(preds)
            #max_items = sorted_preds[(-1 * chord_size):]
            #indices=[]
            #for z in max_items:
            #    indices.append(np.nonzero(preds == z)[0][0])
            indices = np.random.choice(88,chord_size,replace=False, p=[i/sum(preds) for i in preds])
            for index in indices:
                next.append(index)
                rounded_preds[index] = 1
            chord_size_and_preds = np.append(chord_size_zeroed,rounded_preds)
            p=np.array([np.append(p[0],np.array([chord_size_and_preds]),axis=0)])
            p=np.delete(p,0,1)

            nexties.append([pno_pitcher(i) for i in next])

        print()
        # print(nexties)


tot_music = mari_287_to_310[start_index: start_index + maxlen] + nexties

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

midi_generator(notes, 'feldman_dream_1_chord_sizes_random_seed.mid', sustain='yes')

# midi_generator(notes,'feldman_dream_1.mid',sustain = 'no')
