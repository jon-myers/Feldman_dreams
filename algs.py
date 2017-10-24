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

