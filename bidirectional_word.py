# Single Bidirectional LSTM with softmax activation
# Using corpus of Be Here Now only to generate text word by word

# Standard libraries
import pandas as pd
import numpy as np
from datetime import datetime

#import Keras library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint

# Import other libraries
import random
import sys

# Set parameters

# Length of seed text
seed_length = 10

# Number of words to be generated
num_words_gen = 20

# Load in data

# Get path to text
path = 'text/be_here_now.txt'

# Open file
f = open(path, encoding='utf-8')

# Instantiate lines list
text = ''

# Iterate through each line (document), strip newlines, and append to raw_text list
for line in f:
    text += line.rstrip()

# Clean and pre-process text

def clean_text(document_string):
    """
    Function that takes in a document in
    the form of a string, and pre-processes
    it, returning a clean string ready
    to be used to fit a CountVectorizer.

    Pre-processing includes:
    - lower-casing text
    - eliminating punctuation
    - dealing with edge case punctuation
      and formatting
    - replacing contractions with
      the proper full words

    :param: document_string: str

    :returns: cleaned_text: str
    :returns: words: list
    """
    # Make text lowercase
    raw_text = document_string.lower()

    # Replace encoding error with a space
    raw_text = raw_text.replace('\xa0', ' ')

    # Normalize period formatting
    raw_text = raw_text.replace('. ', '.')
    raw_text = raw_text.replace('.', '. ')

    # Replace exclamation point with a space
    raw_text = raw_text.replace('!', ' ')

    # Replace slashes with empty
    raw_text = raw_text.replace('/', '')

    # Replace questin marks with empty
    raw_text = raw_text.replace('??', ' ')
    raw_text = raw_text.replace('?', ' ')

    # Replace dashes with space
    raw_text = raw_text.replace('-', ' ')
    raw_text = raw_text.replace('—', ' ')

    # Replace ... with empty
    raw_text = raw_text.replace('…', '')
    raw_text = raw_text.replace('...', '')

    # Replace = with 'equals'
    raw_text = raw_text.replace('=', 'equals')

    # Replace commas with empty
    raw_text = raw_text.replace(',', '')

    # Replace ampersand with and
    raw_text = raw_text.replace('&', 'and')

    # Replace semi-colon with empty
    raw_text = raw_text.replace(';', '')

    # Replace colon with empty
    raw_text = raw_text.replace(':', '')

    # Get rid of brackets
    raw_text = raw_text.replace('[', '')
    raw_text = raw_text.replace(']', '')

    # Replace parentheses with empty
    raw_text = raw_text.replace('(', '')
    raw_text = raw_text.replace(')', '')

    # Replace symbols with letters
    raw_text = raw_text.replace('$', 's')
    raw_text = raw_text.replace('¢', 'c')

    # Replace quotes with nothing
    raw_text = raw_text.replace('“', '')
    raw_text = raw_text.replace('”', '')
    raw_text = raw_text.replace('"', '')
    raw_text = raw_text.replace("‘", "")

    # Get rid of backslashes indicating contractions
    raw_text = raw_text.replace(r'\\', '')

    # Replace extra spaces with single space
    raw_text = raw_text.replace('   ', ' ')
    raw_text = raw_text.replace('  ', ' ')

    # Some apostrophes are of a different type --> ’ instead of '
    raw_text = raw_text.replace("’", "'")

    # Replace contractions with full words, organized alphabetically
    raw_text = raw_text.replace("can't", 'cannot')
    raw_text = raw_text.replace("didn't", 'did not')
    raw_text = raw_text.replace("doesn't", 'does not')
    raw_text = raw_text.replace("don't", 'do not')
    raw_text = raw_text.replace("hasn't", 'has not')
    raw_text = raw_text.replace("he's", 'he is')
    raw_text = raw_text.replace("i'd", 'i would')
    raw_text = raw_text.replace("i'll", 'i will')
    raw_text = raw_text.replace("i'm", 'i am')
    raw_text = raw_text.replace("isn't", 'is not')
    raw_text = raw_text.replace("it's", 'it is')
    raw_text = raw_text.replace("nobody's", 'nobody is')
    raw_text = raw_text.replace("she's", 'she is')
    raw_text = raw_text.replace("shouldn't", 'should not')
    raw_text = raw_text.replace("that'll", 'that will')
    raw_text = raw_text.replace("that's", 'that is')
    raw_text = raw_text.replace("there'd", 'there would')
    raw_text = raw_text.replace("they're", 'they are')
    raw_text = raw_text.replace("there's", 'there are')
    raw_text = raw_text.replace("we'd", 'we would')
    raw_text = raw_text.replace("we'll", 'we will')
    raw_text = raw_text.replace("we're", 'we are')
    raw_text = raw_text.replace("we've", 'we have')
    raw_text = raw_text.replace("wouldn't", 'would have')
    raw_text = raw_text.replace("you'd", 'you would')
    raw_text = raw_text.replace("you'll", 'you will')
    raw_text = raw_text.replace("you're", 'you are')
    raw_text = raw_text.replace("you've", 'you have')

    # Fix other contractions
    raw_text = raw_text.replace("'s", ' is')

    cleaned_text = raw_text

    # Extract tokens
    text_for_tokens = cleaned_text
    text_for_tokens = text_for_tokens.replace('.', '')
    words = text_for_tokens.split()

    return (cleaned_text, words)

# Call function clean_text to get processed
processed_text, words = clean_text(text)

# Check length of corpus
print('Total words in corpus:', len(processed_text.split()))

# Get list of unique words
tokens = list(set(sorted(words)))
vocab_size = len(tokens)
print('Vocabulary size:', vocab_size)

# Create dictionaries for characters and indices (and vice versa)
word_indices = dict((word, i) for i, word in enumerate(tokens))
indices_word = dict((i, word) for i, word in enumerate(tokens))

# Create list of all sentences in text
sentences = processed_text.split('. ')

# Calculate average length of sentences in words
sentence_lengths = []

for each in sentences:
    sentence_lengths.append(len(each))

avg_length = np.mean(sentence_lengths)

print('The average number of words in a sentence is:', int(round(avg_length)))

# Set sequence length (also known as maxlen)
seq_length = seed_length

# Set steps
step = 1

# Instantiate sequences list containing sequences of words
sequences = []

# Instantiate next_words list containing the next words for each sequence in sequences
next_words = []

# Populate lists
for i in range(0, len(words) - seq_length, step):
    sequences.append(words[i: i + seq_length])
    next_words.append(words[i + seq_length])

print('Sequences:', len(sequences))

print('Vectorization...')

X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)

for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

# Build Model

# Build single Bidirectional LSTM model
print('Build model...')

model = Sequential()
model.add(Bidirectional(LSTM(128), input_shape=(seq_length, vocab_size), merge_mode='concat', weights=None))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

# Choose optimizer
#optimizer = RMSprop(lr=0.01)
#optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Choose loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])

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
    global num_words_gen
    global seed_length

    # Update number of epochs to reflect total number of epochs at end
    epoch_num = epoch + 1

    print()
    print(f'---- Generating text after Epoch: {epoch + 1}')

    # Pick a random starting index to get a seed from sequences
    start_index = random.randint(0, len(sequences) - seq_length - 1)

    print(start_index)

    # For each diversity values
    for diversity in diversity_range:
        print('---- diversity:', diversity)

        # Initialize generated text
        generated = ''

        # Get sentence from list of words using start_index and sequence length
        sentence = words[start_index: start_index + seq_length]

        sentence_str = ' '.join(sentence)
        generated += sentence_str

        print('---- Generating with seed: "' + sentence_str + '"')
        sys.stdout.write(generated)

        # Collect seed text
        seed = generated

        # Initialize collecting of generated text
        curr_str = ''

        for i in range(num_words_gen):
            x_pred = np.zeros((1, seq_length, vocab_size))

            for t, word in enumerate(sentence):
                x_pred[0, t, word_indices[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            # Add a space
            generated += ' '

            # Add the next word to the generated text
            generated += next_word

            # Shift the sentence by one, and add the next character at its end
            sentence = sentence[1:]
            sentence.append(next_word)

            # Output the next word
            sys.stdout.write(' ')
            sys.stdout.write(next_word)
            sys.stdout.flush()

            # Add a space and the next word to our storing of generated text
            curr_str += ' '
            curr_str += next_word

        # Create dict of current diversity / text set and add to list
        results.append(dict([
            ('Epoch', epoch + 1),
            ('Diversity', diversity),
            ('Seed text', seed),
            ('Generated text', curr_str),
            ('Full text', seed + curr_str),
            ('Loss', 0),
            ('Seed', seed_length),
            ('Generated', num_words_gen)
        ]))

        print()
        print()


# Set up callback
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Set number of epochs
epochs = 100

# Fit model
history_callback = model.fit(X, y,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=[print_callback,
                                        ModelCheckpoint(
                                            f'save/word_' + f'{epochs}' + 'bidirectional_beherenow_128.h5py')])

# Set up easy access to loss and categorical accuracy
loss_history = history_callback.history['loss']
cat_acc_history = history_callback.history['categorical_accuracy']

# Get list of each metric N times, N = len(diversity_range), to record metrics for each Epoch
loss_to_add = [loss for loss in loss_history for i in range(len(diversity_range))]
cat_acc_to_add = [acc for acc in cat_acc_history for i in range(len(diversity_range))]

# Add losses to results
for dictionary, curr_loss in zip(results, loss_to_add):
    dictionary['Loss'] = round(curr_loss, 4)

# Add accuracies to results
for dictionary, curr_acc in zip(results, cat_acc_to_add):
    dictionary['Categorical Accuracy'] = round(curr_acc, 4)

# Turn results into dataframe
results = pd.DataFrame(data=results, columns=['Epoch', 'Loss', 'Categorical Accuracy',
                                              'Diversity', 'Seed', 'Generated', 'Seed text',
                                              'Generated text', 'Full text'])

# Export results
results.to_csv(f'results/word_{epoch_num}_bidirectional_beherenow_1step_adam_128_{str(datetime.now())}.csv',
               index=False)