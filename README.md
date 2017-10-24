# Feldman_dreams
Predictive LSTM trained on Morton Feldman's 'Palais de Mari' , for dreaming up new Feldman.

Requires Keras, Tensorflow, NumPy, Pretty-Midi

Works in Python 2.7.10 ; Currently getting buggy  in Python 3

I started from the Keras example lstm_text_generation.py (https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py) 
I would highly recommend getting a handle on that example before diving in to this.

Dream 1 trains on measures 287 - 310 of 'Palais de Mari,' without duration info, just sequences of chords.
1.0 generates the next chord by selecting any predicted notes above 0.1, leading to larger chords than the training data. 
1.1 tries to constrain by making every other note a one pitch, often with comic results. 
1.2 is the same as 1.0, but generates new music based on a random sampling of training data. 
1.3 is the same as 1.0, but generates new music based on random ~44 note chords. 1.4 adds nodes in the LSTM for deciding chord size (in this case, 0, 1, 2, or 3), which gets pretty plausible results.

Dream 2 trains on measures 287 - 310 of 'Palais de Mari,' with nodes for chord size (4) and duration info. Training data is divided into smallest ictus (32nd notes), mostly zeroes. 

Dream 3 trains on page 1 of 'Palais de Mari,' with nodes for chord size (8) and duration info. Training data is divided into smallest ictus (32nd notes), mostly zeroes.

Sorry about the mess!
