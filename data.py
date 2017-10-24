import numpy as np
import math
#format of notes is: [1/4 tone piano pitch, start time, dur, velocity)
import pretty_midi

def midi_generator(notes,file_name,sustain = 'no'):

    notes = sorted(notes, key = lambda x: x[1])
    score=pretty_midi.PrettyMIDI()
    piano_program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano1 = pretty_midi.Instrument(program=piano_program)
    piano2 = pretty_midi.Instrument(program=piano_program)
    piano2.pitch_bends.append(pretty_midi.PitchBend(pitch=-2048,time=0))
    if sustain == 'yes':
        piano1.control_changes.append(pretty_midi.ControlChange(number=64,value=100,time=0))
        piano2.control_changes.append(pretty_midi.ControlChange(number=64,value=100,time=0))
    for n,note_ in enumerate(notes):
        if note_[1] == 0:
            note_[1] = 1e-20
        if note_[0] % 2 == 0:
            for later_note in notes[n+1:]:
                if later_note[0] == note_[0]:
                    if later_note[1] <= note_[1] + note_[2]:
                        note_[2] = 0.9 * (later_note[1] - note_[1])
            note_=pretty_midi.Note(velocity=note_[3],pitch=21+(note_[0]/2),start=note_[1],end=note_[1]+note_[2])
            piano2.notes.append(note_)
        else:
            for later_note in notes[n+1:]:
                if later_note[0] == note_[0]:
                    if later_note[1] <= note_[1] + note_[2]:
                        note_[2] = 0.9 * (later_note[1] - note_[1])
            note_=pretty_midi.Note(velocity=note_[3],pitch=21+((note_[0]-1)/2),start=note_[1],end=note_[1]+note_[2])
            piano1.notes.append(note_)
    score.instruments.append(piano1)
    score.instruments.append(piano2)
    score.write(file_name)


def midi_numberer(pitch):
    if pitch[:-1] == 'c':
        init = 0
    elif pitch[:-1] == 'cs' or pitch[:-1] == 'df':
        init = 1
    elif pitch[:-1] == 'd':
        init = 2
    elif pitch[:-1] == 'ds' or pitch[:-1] == 'ef':
        init = 3
    elif pitch[:-1] == 'e' or pitch[:-1] == 'ff':
        init = 4
    elif pitch[:-1] == 'f' or pitch[:-1] == 'es':
        init = 5
    elif pitch[:-1] == 'fs' or pitch[:-1] == 'gf':
        init = 6
    elif pitch[:-1] == 'g':
        init = 7
    elif pitch[:-1] == 'gs' or pitch[:-1] == 'af':
        init = 8
    elif pitch[:-1] == 'a':
        init = 9
    elif pitch[:-1] == 'bf' or pitch[:-1] == 'as':
        init = 10
    elif pitch[:-1] == 'b':
        init = 11
    else:
        return 'error, bad input!'
    oct = pitch[-1]
    midi_num = 12 + 12*int(oct) + init
    return midi_num

def pno_numberer(pitch):
    if '.' in pitch:
        pitch = pitch.split('.')[0]
    if pitch[:-1] == 'c':
        init = 0
    elif pitch[:-1] == 'cs' or pitch[:-1] == 'df':
        init = 1
    elif pitch[:-1] == 'd':
        init = 2
    elif pitch[:-1] == 'ds' or pitch[:-1] == 'ef':
        init = 3
    elif pitch[:-1] == 'e' or pitch[:-1] == 'ff':
        init = 4
    elif pitch[:-1] == 'f' or pitch[:-1] == 'es':
        init = 5
    elif pitch[:-1] == 'fs' or pitch[:-1] == 'gf':
        init = 6
    elif pitch[:-1] == 'g':
        init = 7
    elif pitch[:-1] == 'gs' or pitch[:-1] == 'af':
        init = 8
    elif pitch[:-1] == 'a':
        init = 9
    elif pitch[:-1] == 'bf' or pitch[:-1] == 'as':
        init = 10
    elif pitch[:-1] == 'b':
        init = 11
    else:
        return 'error, bad input!'
    oct = pitch[-1]
    midi_num = 12 + 12*int(oct) + init - 21
    return midi_num

def grade_and_chorder(set):
    new=[]
    for i, item in enumerate(set):
        if i%2 == 0:
            new.append((item,'g'))
        else:
            new.append((item,4))
    return new
#0 = a0
def pno_pitcher(pno_num):
    pitch=[]
    set=['c','df','d','ef','e','f','gf','g','af','a','bf','b']
    pitch.append(set[(pno_num + 9) % 12])
    pitch.append(str(math.floor((pno_num + 9) / 12 )))
    return ''.join(pitch)


#3 streams: grace, low, high


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


simple_keyboard_mari= np.zeros((len(mari_287_to_310),88))
for i,item in enumerate(simple_pno_mari):
    for j in item:
        simple_keyboard_mari[i][j] = 1.0


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
mari_page_2 = [\
    #43
    (12,[('d4',0),('e4',0),('ef5',0)]),\
    #44
    (16,[('c3',0),('af3',0),('d4',0),('e4',0),('a4',0),('bf4',0),('df5',0),('fs6',15.5)]),\
    #45
    (6,[('f2',0),('e3',0),('ef4',0),('df5',0),('g6',5.5)]),\
    #46
    (16,[]),\
    #47
    (12,[('d3',0),('c4',0),('ef4',0),('df5',0),('f2',6),('e3',6),('ef4',6),('df5',6),('fs6',11.5)]),\
    #48
    (10,[]),\
    #49
    (8,[('af3',0),('bf3',0),('df4',0),('a4',0),('f6',7.5)]),\
    #50
    (6,[]),\
    #51
    (12,[('d3',0),('c4',0),('ef4',0),('df5',0),('g6',11.5)]),\
    #52
    (6,[]),\
    #53
    (12,[('af3',0),('ds4',0),('g4',0),('a5',0)]),\
    #54
    (12,[('bf2',0),('f5',0),('d5',5.5),('g4',6),('c6',6)]),\
    #55
    (8,[('c3',0),('af3',0),('d4',0),('e4',0),('a4',0),('bf4',0),('df5',0),('fs6',15.5),('fs6',7.5)]),\
    #56
    (6,[]),\
    #57
    (12,[('d3',0),('c4',0),('ef4',0),('df5',0),('f2',6),('e3',6),('ef4',6),('df5',6)]),\
    #58
    (6,[]),\
    #59
    (8,[('f2',0),('e3',0),('ef4',0),('df5',0),('g6',7.5)]),\
    #60
    (6,[]),\
    #61
    (16,[('d4',0),('e4',0),('ef5',0)]),\
    #62
    (10, [('c5',0),('af1', 0.5), ('bf4', 0.5), ('g2', 6), ('d4', 6)]),\
    #63
    (10, [('bf2',4),('b6',4)]),\
    #64
    (10,[('a3',0),('af3',0.5),('bf4',0.5),('f4',6),('g4',6.5),('d5',6.5),('d6',6.5)]),\
    #65
    (10,[])  ]
    # FUCKED UP!
    # #66
    # (10,[('ped_up',10),('df3',10),('ef3',10),('gf4',10),('f5',10)]),\
    # #67
    # (16,[('gf4',0),('f5',0),('ped_down',1)]),\
    # #68
    # (6,[('d3',0),('c4',0),('ef4',0),('df5',0)]),\
    # #69
    # (8,[]),\
    # #70
    # (12,[('af3',0),('bf3',0),('df4',0),('a4',0),('f6',11.5)]),\
    # #71
    # (10,[('d4',6),('e4',6),('b4',6),('bf5',6)]),\
    # #72
    # (10,[('ped_up',8)])


def nn_88_keys(score,max_chord):
    beats = sum([i[0] for i in score])
    divs = beats * 2
    simple_pno=[]
    large_thing =[]
    divs_so_far=0
    for i in score:
        for j in i[1]:
            large_thing.append([j[0],(2*j[1])+divs_so_far])
        divs_so_far += (2 * i[0])

    for i in range(divs):
        m=[]
        for j in large_thing:
            if j[1] == i:
                m.append(j[0])
        simple_pno.append(m)





    chord_lens = [len(i) for i in simple_pno]
    chord_lens_nn = np.zeros([divs,max_chord])
    for i,item in enumerate(chord_lens):
        chord_lens_nn[i][item] = 1

    dd=[]
    nn_score = np.zeros([divs,88],bool)
    for i, item in enumerate(simple_pno):
        if len(item) != 0:
            for j in item:
                nn_score[i][pno_numberer(j)] = 1


    nn_tot = np.concatenate((chord_lens_nn, nn_score),axis = 1)
    return nn_tot

a=nn_88_keys(mari_page_1,8)



