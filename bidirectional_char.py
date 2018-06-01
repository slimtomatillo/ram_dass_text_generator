# Single Bidirectional LSTM with softmax activation
# Using corpus of Be Here Now only to generate text character by character

# Standard libraries
import pandas as pd
import numpy as np
import random
import sys
import io
from datetime import datetime

# Keras
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint

# Length of seed text
seed_length = 20

# Number of generated characters to be generated
num_char_gen = 60

# Get data
path = get_file('be_here_now.txt', origin='text/be_here_now.txt')

# Open text, make lowercase
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()

# Clean data - get rid of newlines
tokens = text.split()
text = ' '.join(tokens)

# Check length of corpus
print('corpus length:', len(text))

# Get list of unique characters
chars = sorted(list(set(text)))
print('total characters:', len(chars))

# Create dictionaries for characters and index (and vice versa)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Cut the text into semi-redundant sequences of seq_length characters
seq_length = seed_length
step = 1
sentences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

print('sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), seq_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Build single Bidirectional LSTM model
print('Build model...')
model = Sequential()
model.add(Bidirectional(LSTM(128), input_shape=(seq_length, len(chars)), merge_mode='concat', weights=None))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# Choose optimizer
optimizer = RMSprop(lr=0.01)

# Choose loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Set batch size
batch_size = 128

def sample(preds, temperature=1.0):
    """
    Function to sample an index from
    a probability array.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return(np.argmax(probas))


# Store generated text
results = []

# Get number of epochs
epoch_num = 0

# Set diversity range
diversity_range = [0.3, 0.4, 0.5, 0.6, 0.7]


def on_epoch_end(epoch, logs):
    """
    Function invoked at end of each Epoch.
    Prints generated text.
    """
    global results
    global epoch_num
    global diversity_range
    global num_char_gen
    global seed_length

    # Update number of epochs to reflect total number of epochs at end
    epoch_num = epoch + 1

    print()
    print(f'---- Generating text after Epoch: {epoch + 1}')

    start_index = random.randint(0, len(text) - seq_length - 1)
    for diversity in diversity_range:
        print('---- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + seq_length]

        generated += sentence

        print('---- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        # Collect seed text
        seed = generated

        # Initialize collecting of generated text
        curr_str = ''

        for i in range(num_char_gen):
            x_pred = np.zeros((1, seq_length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char

            print('SENTENCE', sentence)
            print('SENTENCE-1:', sentence[1:])
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

            curr_str += next_char

        # Create dict of current diversity / text set and add to list
        results.append(dict([
            ('Epoch', epoch + 1),
            ('Diversity', diversity),
            ('Seed text', seed),
            ('Generated text', curr_str),
            ('Full text', seed + curr_str),
            ('Loss', 0),
            ('Seed', seed_length),
            ('Generated', num_char_gen)
        ]))

        print()
        print()

# Set up callback
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Set number of epochs
epochs = 100

# Fit model
history_callback = model.fit(x, y,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=[print_callback,
                                        ModelCheckpoint(
                                            f'save/char_' + f'{epochs}' + 'bidirectional_beherenow_128.h5py')])

# Set up easy access to loss
loss_history = history_callback.history['loss']

# Get list of each loss N times, N = len(diversity_range), to record losses for each Epoch
loss_to_add = [loss for loss in loss_history for i in range(len(diversity_range))]

# Add losses to results
for dictionary, curr_loss in zip(results, loss_to_add):
    dictionary['Loss'] = round(curr_loss, 4)

# Turn results into dataframe
results = pd.DataFrame(data=results, columns=['Epoch', 'Loss', 'Diversity',
                                              'Seed', 'Generated', 'Seed text',
                                              'Generated text', 'Full text'])

# Export results
results.to_csv(f'results/char_{epoch_num}_bidirectional_beherenow_1step_128_{str(datetime.now())}.csv', index=False)