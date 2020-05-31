# ========================================================================
# ========================================================================
# ChoGAN Utility Functions
# ========================================================================
# ========================================================================
# --------------------------------------------------------
# Load external packages
# --------------------------------------------------------
# General packages
import os
import pandas as pd
import numpy as np
import math
import pickle
import glob
import warnings

# Reading/writing MIDI files
from mido import MidiFile
from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream, duration
import pretty_midi

# Data transformation
import datetime
import time
from time import time, strftime
from sklearn.utils import shuffle

# Data visualization
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Tensorflow
from tensorflow.keras.layers import LSTM, GRU, Input, Dropout, Dense, Activation, \
    Embedding, Concatenate, Reshape, Bidirectional, RepeatVector, Permute, \
    Multiply, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow.keras.backend as K 
from tensorflow.python.framework import ops
from tensorflow.keras import utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# Define utility functions
# --------------------------------------------------------
# Collect MIDI information
def collect_midi_data(parent_dir, seed=105):
    file_list = []
    for p, s, f in os.walk(parent_dir):
        for n in f:
            file_list.append(os.path.join(p, n))
    print('Imported', len(file_list), 'files.')
    midi_data = pd.DataFrame({'file': file_list})
    midi_data = midi_data[midi_data['file'].str.contains('DS_Store') == False].reset_index(drop=True)
    
    midi_data = shuffle(midi_data, random_state=seed).reset_index(drop=True)

    midi_data['midi'] = None
    midi_data['tempo'] = None
    midi_data['n_seconds'] = None
    for i in range(len(midi_data)):
        midi_data['midi'][i] = pretty_midi.PrettyMIDI(midi_data['file'][i])
        midi_data['tempo'][i] =  midi_data['midi'][i].estimate_tempo()
        midi_data['n_seconds'][i] =  midi_data['midi'][i].get_end_time()

    midi_data['n_seconds'] = midi_data['n_seconds'].astype(int)
    midi_data['n_minutes'] = midi_data['n_seconds'] // 60 + round((midi_data['n_seconds'] % 60)/100, 2)

    return midi_data

# Return song metrics
def song_metrics(data):
    print('Number of songs:', len(data))
    print('Total hours of music:', round(np.sum(data['n_seconds'])/60/60, 1))
    print('Average song duration (in minutes):', round(np.mean(data['n_minutes']), 1))
    
 # Plot distributions   
def plot_dists(data, title):
    fig, axs = plt.subplots(ncols=2, figsize=(13, 5))
    sns.distplot(data, ax=axs[0], color='cadetblue') \
            .set(xlabel=title)
    sns.boxplot(data, ax=axs[1], color='skyblue') \
        .set(xlabel=title)
    return plt.show()

# Extract notes from MIDI files
def extract_notes(data):
    df = data.copy()
    for f in range(len(df)):
        notes = []
        durations = []
        score = converter.parse(df['file'][f]).chordify()
        for element in score.flat:
            if isinstance(element, note.Note):
                if element.isRest:
                    notes.append('rest')
                    durations.append(element.duration.quarterLength)
                else:
                    notes.append(str(element.nameWithOctave))
                    durations.append(element.duration.quarterLength)

            if isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element.semiClosedPosition().pitches))
                durations.append(element.duration.quarterLength)

        df['notes'][f] = notes
        df['durations'][f] = durations
    df['n_notes'] = [len(df['notes'][i]) for i in range(len(df))]
    return df

# Create note embeddings
def create_note_embeddings(notes, note_names):
    # TF - Note Term Frequencies 
    notes_df = pd.DataFrame({'note':notes, 
                             'tf':1}) \
                    .groupby('note') \
                    .agg({'tf':sum}) \
                    .reset_index()

    # Split chords into indivudal notes (i.e., A2.E3 to A2 and E3)
    unique_notes = list(set(np.concatenate(notes_df['note'].str.split('.'))))

    key_note_map = pd.concat(
        [pd.DataFrame({'note':note_names}),
         pd.DataFrame(np.zeros((len(note_names),
                                len(unique_notes))), columns=unique_notes)], axis=1)

    key_note_map['note_split'] =  key_note_map['note'].str.split('.')
    key_note_map['tf'] = notes_df['tf'] / sum(notes_df['tf'])

    for r in range(len(key_note_map)):
        for n in key_note_map['note_split'][r]:
            if n in unique_notes:
                key_note_map[n][r] = key_note_map['tf'][r] 

    key_note_map = key_note_map.drop(['note_split', 'tf'], axis=1)
    
    key_note_emb = np.asarray(key_note_map.copy().drop('note', axis=1))

    print('Note embedding dimensions:', key_note_emb.shape)
    return key_note_emb


# Distinct notes and durations
def get_distinct(elements):
    # Get all pitch names
    element_names = sorted(set(elements))
    n_elements = len(element_names)
    return (element_names, n_elements)


# Lookups for notes and durations
def create_lookups(element_names):
    # create dictionary to map notes and durations to integers
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))
    return (element_to_int, int_to_element)


# Prepare sequqnces based on sequence lengths
def prepare_sequences(notes, durations, lookups, distincts, seq_len):
    """ Prepare the sequences used to train the Neural Network """

    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups
    note_names, n_notes, duration_names, n_durations = distincts

    notes = (['sos']*(seq_len-1))+notes
    durations = ([0]*(seq_len-1))+durations
    
    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - seq_len):
        notes_sequence_in = notes[i:i + seq_len]
        notes_sequence_out = notes[i + seq_len]
        notes_network_input.append([note_to_int[char] for char in notes_sequence_in])
        notes_network_output.append(note_to_int[notes_sequence_out])

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    n_patterns = len(notes_network_input)

    notes_network_input = np.reshape(notes_network_input, (n_patterns, seq_len))
    notes_network_input = notes_network_input/float(n_notes)
    
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))
    durations_network_input = durations_network_input/float(n_durations)
    
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = utils.to_categorical(notes_network_output, num_classes=n_notes)
    durations_network_output = utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return (network_input, network_output)

# Get sequence intputs and outputs 
def get_input_outputs(data, lookups, distincts, seq_len):
    in_array, out_array = prepare_sequences(
        list(np.concatenate(data.padded_notes)), 
        list(np.concatenate(data.padded_durations)), 
        lookups, distincts, seq_len)
    return (in_array, out_array)

# LSTM model
def create_lstm_network(n_notes, n_durations, key_note_emb):
    ops.reset_default_graph()
    """ create the structure of the neural network """
    n_embeddings = key_note_emb.shape[1]
    notes_in = Input(shape = (None,))
    durations_in = Input(shape = (None,))

    x1 = Embedding(n_notes, n_embeddings,  weights = [key_note_emb], name='pitch_embeddings')(notes_in)
    x2 = Embedding(n_durations, 10, name='duration_embeddings')(durations_in) 

    x = Concatenate()([x1,x2])
    
    # First hidden layer
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    
    # Second hidden layer
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    # Attention mechanism 
    e = Dense(1, activation='tanh')(x)
    e = Reshape([-1])(e)
    alpha = Activation('softmax')(e)
    alpha_repeated = Permute([2, 1])(RepeatVector((128))(alpha))
    c = Multiply()([x, alpha_repeated])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(128,))(c)
    
    # Output layer
    notes_out = Dense(n_notes, activation = 'softmax', name = 'pitch')(c)
    durations_out = Dense(n_durations, activation = 'softmax', name = 'duration')(c)
   
    model = Model([notes_in, durations_in], [notes_out, durations_out])

    attn_model = Model([notes_in, durations_in], alpha)

    opti = RMSprop(lr = 0.001, clipnorm=1.)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model, attn_model

# GRU model
def create_gru_network(n_notes, n_durations, key_note_emb):
    ops.reset_default_graph()
    """ create the structure of the neural network """
    n_embeddings = key_note_emb.shape[1]
    notes_in = Input(shape = (None,))
    durations_in = Input(shape = (None,))

    x1 = Embedding(n_notes, n_embeddings,  weights = [key_note_emb], name='pitch_embeddings')(notes_in)
    x2 = Embedding(n_durations, 10, name='duration_embeddings')(durations_in) 

    x = Concatenate()([x1,x2])
    
    # First hidden layer
    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    
    # Second hidden layer
    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    # Attention mechanism 
    e = Dense(1, activation='tanh')(x)
    e = Reshape([-1])(e)
    alpha = Activation('softmax')(e)
    alpha_repeated = Permute([2, 1])(RepeatVector((128))(alpha))
    c = Multiply()([x, alpha_repeated])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(128,))(c)
    
    notes_out = Dense(n_notes, activation = 'softmax', name = 'pitch')(c)
    durations_out = Dense(n_durations, activation = 'softmax', name = 'duration')(c)
   
    model = Model([notes_in, durations_in], [notes_out, durations_out])

    attn_model = Model([notes_in, durations_in], alpha)

    opti = RMSprop(lr = 0.001, clipnorm=1.)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model, attn_model

# LSTM + GRU model
def create_gru_lstm_network(n_notes, n_durations, key_note_emb):
    ops.reset_default_graph()
    """ create the structure of the neural network """
    n_embeddings = key_note_emb.shape[1]
    notes_in = Input(shape = (None,))
    durations_in = Input(shape = (None,))

    x1 = Embedding(n_notes, n_embeddings,  weights = [key_note_emb], name='pitch_embeddings')(notes_in)
    x2 = Embedding(n_durations, 10, name='duration_embeddings')(durations_in) 

    x = Concatenate()([x1,x2])
    
    # First hidden layer
    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    # Attention mechanism 
    e = Dense(1, activation='tanh')(x)
    e = Reshape([-1])(e)
    alpha = Activation('softmax')(e)
    alpha_repeated = Permute([2, 1])(RepeatVector((128))(alpha))
    c = Multiply()([x, alpha_repeated])
    c = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(128,))(c)
    
    notes_out = Dense(n_notes, activation = 'softmax', name = 'pitch')(c)
    durations_out = Dense(n_durations, activation = 'softmax', name = 'duration')(c)
   
    model = Model([notes_in, durations_in], [notes_out, durations_out])

    attn_model = Model([notes_in, durations_in], alpha)

    opti = RMSprop(lr = 0.001, clipnorm=1.)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model, attn_model

# Train model
def train_network(model, exp_folder, seq_set):
    
    train_in = seq_set[0]
    train_out = seq_set[1]
    test_in = seq_set[2]
    test_out = seq_set[3]

    batch_size=256
    n_epochs = math.ceil(train_in[0].shape[0]/batch_size)

    print('Training over', n_epochs, 'epochs.\n')

    # Store weights
    weights_folder = 'storage/experiments/'+exp_folder

    # Start time
    start_time = time()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        
        checkpoint = ModelCheckpoint(
            os.path.join(weights_folder, "weights.h5"),
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min')

        early_stopping = EarlyStopping(
            monitor='loss',
            restore_best_weights=True,
            patience = 10)


        callbacks_list = [
            checkpoint,
            early_stopping]

        # Save model
        model.save_weights(os.path.join(weights_folder, 'weights.h5'))
        history = model.fit(train_in, train_out,
                            epochs=n_epochs, 
                            batch_size=batch_size,
                            validation_data=(test_in, test_out), 
                            callbacks=callbacks_list, verbose=False)
        
    
    model_time = round((time() - start_time), 0)
    print(len(history.history['loss']), 'epochs completed in', model_time, 'seconds.\n')
    return history

# Plot loss
def plot_loss(exp):
    # Summarize history for loss
    plt.plot(exp.history['loss'])
    plt.plot(exp.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    return plt.show()

# Evaluate loss
def evaluate_loss(model, exp, seq_set):
    train_in = seq_set[0]
    train_out = seq_set[1]
    test_in = seq_set[2]
    test_out = seq_set[3]
    
    train_loss = model.evaluate(train_in, train_out)
    test_loss = model.evaluate(test_in, test_out)
    
    # Store results
    results = pd.DataFrame(
        {'Set': ['Train', 'Test'], 
         'Loss': [train_loss[0], test_loss[0]], 
         'Pitch Loss': [train_loss[1], test_loss[1]], 
         'Duration Loss': [train_loss[2], test_loss[2]]})
    
    results['Loss'] = round(results['Loss'], 4)
    results['Pitch Loss'] = round(results['Pitch Loss'], 4)
    results['Duration Loss'] = round(results['Duration Loss'], 4)
    results.to_csv('storage/outputs/loss_metrics/'+exp+'_train_val_loss.csv', index=False)
    return results


# Generate notes
def generate_notes(model, attn_model, notes_temp, duration_temp, 
    max_extra_notes, max_seq_len, seq_len, note_to_int,  int_to_note, 
    duration_to_int, int_to_duration):

    notes = ['sos']
    durations = [0]

    if seq_len is not None:
        notes = ['sos'] * (seq_len - len(notes)) + notes
        durations = [0] * (seq_len - len(durations)) + durations

    sequence_length = len(notes)

    def sample_with_temp(preds, temperature):
        if temperature == 0:
            return np.argmax(preds)
        else:
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            return np.random.choice(len(preds), p=preds)

    prediction_output = []
    notes_input_sequence = []
    durations_input_sequence = []

    overall_preds = []

    for n, d in zip(notes,durations):
        note_int = note_to_int[n]
        duration_int = duration_to_int[d]

        notes_input_sequence.append(note_int)
        durations_input_sequence.append(duration_int)

        prediction_output.append([n, d])

        if n != 'sos':
            midi_note = note.Note(n)
            new_note = np.zeros(128)
            new_note[midi_note.pitch.midi] = 1
            overall_preds.append(new_note)

        attn_matrix = np.zeros(shape = (max_extra_notes+sequence_length, max_extra_notes))



    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for note_index in range(max_extra_notes):

            prediction_input = [
                np.array([notes_input_sequence]),
                np.array([durations_input_sequence])
               ]

            notes_prediction, durations_prediction = model.predict(prediction_input, verbose=0)
            attn_prediction = attn_model.predict(prediction_input, verbose=0)[0]
            attn_matrix[(note_index-len(attn_prediction)+sequence_length):(note_index+sequence_length), note_index] = attn_prediction

            new_note = np.zeros(128)

            for idx, n_i in enumerate(notes_prediction[0]):
                try:
                    note_name = int_to_note[idx]
                    midi_note = note.Note(note_name)
                    new_note[midi_note.pitch.midi] = n_i
                except:
                    pass

            overall_preds.append(new_note)


            i1 = sample_with_temp(notes_prediction[0], notes_temp)
            i2 = sample_with_temp(durations_prediction[0], duration_temp)


            note_result = int_to_note[i1]
            duration_result = int_to_duration[i2]

            prediction_output.append([note_result, duration_result])

            notes_input_sequence.append(i1)
            durations_input_sequence.append(i2)

            if len(notes_input_sequence) > max_seq_len:
                notes_input_sequence = notes_input_sequence[1:]
                durations_input_sequence = durations_input_sequence[1:]

            if note_result == 'sos':
                break

        overall_preds = np.transpose(np.array(overall_preds)) 
        print('Generated sequence of {} notes'.format(len(prediction_output)))
    return prediction_output


# Store generated MIDI 
def store_generated_midi(prediction_output, exp):

    midi_stream = stream.Stream()

    # Create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        note_pattern, duration_pattern = pattern

        # If pattern is a chord
        if ('.' in note_pattern):
            notes_in_chord = note_pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.duration = duration.Duration(duration_pattern)
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            midi_stream.append(new_chord)

        # If pattern is a rest   
        elif note_pattern == 'rest':
            new_note = note.Rest()
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Piano()
            midi_stream.append(new_note)

        # If note is not sos / start of song 
        elif note_pattern != 'sos':
            new_note = note.Note(note_pattern)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Piano()
            midi_stream.append(new_note)

    midi_stream = midi_stream.chordify()
    timestr = strftime("%Y%m%d-%H%M%S")
    midi_stream.write('midi', 
        fp=os.path.join('storage/outputs/gen_midis', exp +'_output-' + timestr + '.mid'))
