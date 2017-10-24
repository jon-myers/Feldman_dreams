# with chord info as first items in neural net [1,0,0,0,0,0] means chord size of 0, [0,1,0,0,0,0] means chord size of 1, etc.

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
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

mari_pg_8_partial = []
counter = 0
for i,item in enumerate(mari_287_to_310):
    if i % 2 == 0:
        next=[]
        next.append(item)
    else:
        next.append(item)
        mari_pg_8_partial.append(next)
def expander(set):
    g=[(i,0.5) for i in set]
    return g
mari_pg_8_partial = [(16, [(i[0][0],0)] + expander(i[1])) for i in mari_pg_8_partial]

tot_length = len(mari_pg_8_partial) * 16 * 2
simple_pno_mari = []
for i in mari_pg_8_partial:
    for j in range(32):
        next=[]
        for k in i[1]:
            if k[1] == j/2.0:
                next.append(pno_numberer(k[0]))
        simple_pno_mari.append(next)









simple_keyboard_mari= np.zeros((len(simple_pno_mari),88),dtype=bool)
for i,item in enumerate(simple_pno_mari):
    for j in item:
        simple_keyboard_mari[i][j] = 1.0

largest_chord = 4
chord_size=[len(i) for i in simple_pno_mari]
chord_size_nn=np.zeros([len(chord_size),largest_chord],bool)
for i,item in enumerate(chord_size):
    chord_size_nn[i][item] = 1

simple_keyboard_mari=np.concatenate((chord_size_nn,simple_keyboard_mari),axis=1)



print('score length:', len(simple_keyboard_mari))

# cut the text in semi-redundant sequences of maxlen characters


maxlen = 32 * 5
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
model.add(LSTM(96, input_shape=(maxlen, 88 + largest_chord)))
model.add(Dense(88 + largest_chord))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='poisson', optimizer=optimizer)



# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    model.fit(X, y,
              batch_size=128,
              epochs=200)

    start_index = random.randint(0, len(simple_keyboard_mari) - maxlen - 1)

    for diversity in [1.0]:
        print()

        generated = ''
        phrase = str(simple_pno_mari[start_index: start_index + maxlen])
        p=np.array([simple_keyboard_mari[start_index: start_index + maxlen]])
        generated += phrase
        print('----- Generating with seed: "' + phrase + '"')
        nexties = []

        for i in range(1500):
            if str(i)[-1] == '0':
                print(str(i))


            preds = model.predict(p, verbose=0)[0]
            chord_size_preds=preds[:largest_chord]
            # chord_size = np.nonzero(chord_size_preds == max(chord_size_preds))[0][0]
            chord_size = np.random.choice(len(chord_size_preds),1,p=[i / sum(chord_size_preds) for i in chord_size_preds])[0]
            chord_size_zeroed = np.zeros(largest_chord,bool)
            chord_size_zeroed[chord_size] = 1
            preds = preds[largest_chord:]
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
simple_pno_mari = simple_pno_mari[start_index:start_index+maxlen]
simple_pno_mari=[[pno_pitcher(i[j]) for j in range(len(i))] for i in simple_pno_mari]

tot_music = simple_pno_mari + nexties



notes=[]
time=0.0
for i in tot_music:
    dur = 2.0 /16.0
    for j in i:
        notes.append([pno_numberer(j)*2,time,dur,30])
    time += dur

midi_generator(notes, 'feldman_dream_2.4.mid', sustain='yes')

