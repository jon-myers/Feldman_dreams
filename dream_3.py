# with chord info as first items in neural net [1,0,0,0,0,0] means chord size of 0, [0,1,0,0,0,0] means chord size of 1, etc.

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import pretty_midi
from data import midi_numberer, pno_numberer, grade_and_chorder, pno_pitcher, nn_88_keys
from algs import midi_generator

mari_page_1 = [\
    # 1
    (10,[('af4',0),('f4',2),('ds5',6),('e4',8)]),\
    # 2
    (12,[]),\
    # 3
    (10,[('af4',0),('f4',2),('e4',5),('ds5',6)]),\
    # 4
    (12,[]),\
    # 5
    (10, [('af4', 0), ('f4', 2), ('ds5', 4), ('e4', 8)]), \
    # 6
    (12,[]),\
    # 7
    (10,[('d5',0),('e5',2),('d5',5),('ds5',6)]),\
    # 8
    (10, [('d5', 0), ('e5', 2), ('d5', 5), ('ds5', 6)]), \
    # 9
    (12,[]),\
    # 10
    (10, [('d5', 0), ('e5', 2), ('d5', 5), ('ds5', 6)]), \
    # 11
    (10, [('d5', 0), ('e5', 2), ('d5', 5), ('ds5', 6)]), \
    # 12
    (10, [('d5', 0), ('e5', 2), ('d5', 5), ('ds5', 6)]), \
    # 13
    (12,[]),\
    # 14
    (10,[('af3',0),('f3',2),('e3',5),('ds4',6)]),\
    #15
    (16,[]),\
    #16
    (10, [('af3', 0), ('f3', 4), ('ds4', 6), ('e3', 7)]), \
    #17
    (20,[]),\
    #18
    (10,[('af2',0),('bf5',0),('g3',6),('d5',6)]),\
    #19
    (16,[]),\
    #20
    (10, [('bf3', 0), ('f6', 0), ('c5', 6), ('g6', 6)]),\
    #21
    (12,[]),\
    #22
    (10,[('af1',0),('bf4',0),('g2',4),('d4',4)]),\
    #23
    (12,[]),\
    #24
    (16,[('g4',0),('a4',0),('ds5',0),('af5',0)]),\
    #25
    (10,[('a3',2)]),\
    #26
    (10,[('c6',6)]),\
    #27
    (12,[]),\
    #28
    (10,[('bf2',0),('f4',0),('c6',6),('g6',6)]),\
    #29
    (10,[('bf4',4),('b6',4)]),\
    #30
    (10, [('af5',0),('f5',2),('e5',5),('ds6',6)]),\
    #31
    (10,[]),\
    #32
    (10,[('bf3',0),('f5',0),('g4',6),('c6',6)]),\
    #33
    (10,[]),\
    #34
    (16,[('af3',0),('ds4',0),('g4',0),('a5',0)]),\
    #35
    (16,[]),\
    #36
    (12,[('f2',0),('e3',0),('ef4',0),('df5',0),('g6',11.5)]),\
    #37
    (6,[('c3',0),('af3',0),('d4',0),('e4',0),('a4',0),('bf4',0),('df5',0),('fs6',5.5)]),\
    #38
    (16,[]),\
    #39
    (12,[('af3',0),('bf3',0),('df4',0),('a4',0),('g6',11.5)]),\
    #40
    (10,[]),\
    #41
    (16,[('d3',0),('c4',0),('ef4',0),('df5',0),('fs6',15.5)]),\
    #42
    (10,[]) ]

#
# mari_287_to_310 = [['d4'],['g4','c5'],['f4'],['af3','ef5'],['d4'],['g4','c5'],['f4'],['af3','ef5'],['f4'],['g5','ef5'],\
#                    ['df4'],['c6','ef5'],['f4'],['bf3','ef4'],['d4'],['gs5','a5','ef5'],['d4'],['b3','c5'],['ef4'],['c6','df5'],\
#                    ['f5'],['bf4','ef6'],['df5'],['d4','c6'],['ef4'],['af4','df5'],['fs4'],['a3','e5'],['ef4'],['af4','df5'],\
#                    ['fs4'],['a3','e5'],['fs4'],['af5','e5'],['d4'],['cs6','e5'],['fs4'],['b3','e5'],['ds4'],['a5','bf5','e5'],['ds4'],['c4','cs5'],\
#                    ['e4'],['cs6','d5'],['fs5'],['b4','e6'],['d5'],['ef4','cs6']]
#
# mari_pg_8_partial = []
# counter = 0
# for i,item in enumerate(mari_287_to_310):
#     if i % 2 == 0:
#         next=[]
#         next.append(item)
#     else:
#         next.append(item)
#         mari_pg_8_partial.append(next)
# def expander(set):
#     g=[(i,0.5) for i in set]
#     return g
# mari_pg_8_partial = [(16, [(i[0][0],0)] + expander(i[1])) for i in mari_pg_8_partial]
#
# tot_length = len(mari_pg_8_partial) * 16 * 2
# simple_pno_mari = []
# for i in mari_pg_8_partial:
#     for j in range(32):
#         next=[]
#         for k in i[1]:
#             if k[1] == j/2.0:
#                 next.append(pno_numberer(k[0]))
#         simple_pno_mari.append(next)
#
#
#
#





# simple_keyboard_mari= np.zeros((len(simple_pno_mari),88),dtype=bool)
# for i,item in enumerate(simple_pno_mari):
#     for j in item:
#         simple_keyboard_mari[i][j] = 1.0
#
# largest_chord = 4
# chord_size=[len(i) for i in simple_pno_mari]
# chord_size_nn=np.zeros([len(chord_size),largest_chord],bool)
# for i,item in enumerate(chord_size):
#     chord_size_nn[i][item] = 1
#
# simple_keyboard_mari=np.concatenate((chord_size_nn,simple_keyboard_mari),axis=1)
largest_chord = 8
simple_keyboard_mari = nn_88_keys(mari_page_1,largest_chord)


print('score length:', len(simple_keyboard_mari))

# cut the text in semi-redundant sequences of maxlen characters


maxlen = 32 * 4
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
model.add(LSTM(88 + largest_chord, input_shape=(maxlen, 88 + largest_chord)))
model.add(Dense(88 + largest_chord))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='poisson', optimizer=optimizer)



# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=64,
              epochs=200)
    model.save('dream_3.h5')


    start_index = random.randint(0, len(simple_keyboard_mari) - maxlen - 1)

    for diversity in [1.0]:
        print()

        generated = ''
        # phrase = str(simple_pno_mari[start_index: start_index + maxlen])
        p=np.array([simple_keyboard_mari[start_index: start_index + maxlen]])
        # generated += phrase
        # print('----- Generating with seed: "' + phrase + '"')
        nexties = []

        for i in range(3000):
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
# simple_pno_mari = simple_pno_mari[start_index:start_index+maxlen]
# simple_pno_mari=[[pno_pitcher(i[j]) for j in range(len(i))] for i in simple_pno_mari]

tot_music = nexties



notes=[]
time=0.0
for i in tot_music:
    dur = 2.0 /16.0
    for j in i:
        notes.append([pno_numberer(j)*2,time,dur,30])
    time += dur

midi_generator(notes, 'feldman_dream_3.3.again.mid', sustain='yes')

