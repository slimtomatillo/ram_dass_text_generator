# Use text generation models to generate text given a seed of the user's input

# Import libraries
import numpy as np
import io

# Keras
from keras.utils.data_utils import get_file
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM

# Logging
import logging
logging.basicConfig(level=logging.INFO)

def generate(user_input):

    # Make user's input seed text
    seed = user_input.lower()

    # Pad user's input if less than 20 char
    while len(seed) < 20:
        seed = ' ' + seed

    # Print seed text entered by user above
    print('Generating text with seed:', user_input)

    # Number of characters to be generated
    num_char_gen = 60

    # Get data
    #path = get_file('be_here_now.txt', origin='text/be_here_now.txt')
    path = get_file('full_text.txt', origin='text/full_text.txt')

    # Open text, make lowercase
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()

    # Clean data - get rid of newlines
    tokens = text.split()
    text = ' '.join(tokens)

    # Check length of corpus
    logging.info(f'Corpus length: {len(text)}')

    # Get list of unique characters
    chars = sorted(list(set(text)))
    logging.info(f'Total characters: {len(chars)}')

    # Create dictionaries for characters and index (and vice versa)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # Cut the text into semi-redundant sequences of seq_length characters
    seq_length = 20
    step = 1
    sentences = []
    next_chars = []

    for i in range(0, len(text) - seq_length, step):
        sentences.append(text[i: i + seq_length])
        next_chars.append(text[i + seq_length])

    logging.info(f'Sequences: {len(sentences)}')

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_length, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # Load the network weights
    filename = 'save/char_10_lstm_full_128.h5py'
    model.load_weights(filename)

    # Instantiate generated text string
    generated = ''

    for i in range(num_char_gen):

        # Create the vector
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(seed):
            x_pred[0, t, char_indices[char]] = 1.

        # Calculate next character
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, 0.34)
        next_char = indices_char[next_index]

        # Add the next character to the text
        generated += next_char

        # Shift the sentence by one, and add the next character at its end
        seed = seed[1:] + next_char

    return(user_input + generated + '.')

def sample(preds, temperature=1.0):
    """
    Helper function to sample an index
    from a probability array.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return(np.argmax(probas))
