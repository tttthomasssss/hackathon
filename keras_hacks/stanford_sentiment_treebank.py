__author__ = 'thomas'
import os

from common import dataset_utils
from common import paths
from keras.datasets import imdb
from keras.datasets import reuters
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np


def run_keras_cnn_example():
	# set parameters:
	max_features = 5000
	maxlen = 100
	batch_size = 32
	embedding_dims = 100
	nb_filter = 250
	filter_length = 3
	hidden_dims = 250
	nb_epoch = 2

	print('Loading data...')
	(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
														  test_split=0.2)
	print(len(X_train), 'train sequences')
	print(len(X_test), 'test sequences')

	print('Pad sequences (samples x time)')
	X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	print('Build model...')
	model = Sequential()

	# we start off with an efficient embedding layer which maps
	# our vocab indices into embedding_dims dimensions
	model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
	model.add(Dropout(0.25))

	# we add a Convolution1D, which will learn nb_filter
	# word group filters of size filter_length:
	model.add(Convolution1D(nb_filter=nb_filter,
							filter_length=filter_length,
							border_mode='valid',
							activation='tanh',
							subsample_length=1))
	# we use standard max pooling (halving the output of the previous layer):
	model.add(MaxPooling1D(pool_length=2))

	# We flatten the output of the conv layer,
	# so that we can add a vanilla dense layer:
	model.add(Flatten())

	# We add a vanilla hidden layer:
	model.add(Dense(hidden_dims))
	model.add(Dropout(0.25))
	model.add(Activation('tanh'))

	# We project onto a single unit output layer, and squash it with a sigmoid:
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='rmsprop',
				  class_mode='binary')
	model.fit(X_train, y_train, batch_size=batch_size,
			  nb_epoch=nb_epoch, show_accuracy=True,
			  validation_data=(X_test, y_test))

def run_keras_example():
	max_words = 1000
	batch_size = 32
	nb_epoch = 5

	print('Loading data...')
	(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
	print(len(X_train), 'train sequences')
	print(len(X_test), 'test sequences')

	nb_classes = np.max(y_train)+1
	print(nb_classes, 'classes')

	print('Vectorizing sequence data...')
	tokenizer = Tokenizer(nb_words=max_words)
	X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
	X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	print('Y_train shape:', Y_train.shape)
	print('Y_test shape:', Y_test.shape)

	print('Building model...')
	model = Sequential()
	model.add(Dense(512, input_shape=(max_words,)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam')

	history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
	score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])



if (__name__ == '__main__'):
	#run_keras_example()
	#run_keras_cnn_example()

	data = dataset_utils.fetch_stanford_sentiment_treebank_dataset()

	y_train, y_valid, y_test = data[1], data[3], data[5]
	#X_train, X_valid, X_test = data[0], data[2], data[4]

	vec = CountVectorizer()
	X_train = vec.fit_transform([' '.join(l) for l in data[0]])
	X_valid = vec.transform([' '.join(l) for l in data[2]])
	X_test = vec.transform([' '.join(l) for l in data[4]])

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts([' '.join(l) for l in data[0]])

	X_train_keras = tokenizer.texts_to_sequences([' '.join(l) for l in data[0]])
	X_test_keras = tokenizer.texts_to_sequences([' '.join(l) for l in data[4]])
	X_valid_keras = tokenizer.texts_to_sequences([' '.join(l) for l in data[2]])
	X_train_keras = tokenizer.sequences_to_matrix(X_train_keras)
	X_test_keras = tokenizer.sequences_to_matrix(X_test_keras)
	X_valid_keras = tokenizer.sequences_to_matrix(X_valid_keras)

	n_classes = np.max(y_train) + 1

	Y_train = np_utils.to_categorical(y_train, n_classes)
	Y_test = np_utils.to_categorical(y_test, n_classes)
	Y_valid = np_utils.to_categorical(y_valid, n_classes)

	print('KERAS...')
	### MLP
	model = Sequential()
	model.add(Dense(output_dim=2048, input_dim=X_test_keras.shape[1], init='glorot_normal', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
	model.add(Activation('tanh'))
	model.add(Dense(output_dim=256, input_dim=2048, init='glorot_normal', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
	model.add(Activation('tanh'))
	model.add(Dense(output_dim=n_classes, init='glorot_normal'))
	model.add(Activation('softmax'))
	### LSTM
	#model = Sequential()
	#model.add(Embedding(X_train.shape[1], 100))
	#model.add(LSTM(100))
	#model.add(Dropout(0.5))
	#model.add(Dense(5))
	#model.add(Activation('sigmoid'))
	### CNN
	#max_features = 5000
	#maxlen = 100
	#batch_size = 32
	#embedding_dims = 100
	#nb_filter = 250
	#filter_length = 3
	#hidden_dims = 250
	#nb_epoch = 2
	#
	#X_train_keras = sequence.pad_sequences(X_train_keras, maxlen=maxlen)
	#X_test_keras = sequence.pad_sequences(X_test_keras, maxlen=maxlen)
	#
	#model = Sequential()
	#model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
	#model.add(Dropout(0.25))
	#model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='tanh', subsample_length=1))
	#model.add(MaxPooling1D(pool_length=2))
	#model.add(Flatten())
	#model.add(Dense(hidden_dims))
	#model.add(Dropout(0.25))
	#model.add(Activation('tanh'))
	#model.add(Dense(5))
	#model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	model.fit(X_train_keras, Y_train, nb_epoch=10, batch_size=50, show_accuracy=True, validation_data=(X_valid_keras, Y_valid))

	score = model.evaluate(X_test_keras, Y_test, batch_size=50, show_accuracy=True)

	print('Objective Score: {}; Accuracy={}'.format(score[0], score[1]))

	print('MNB...')
	mnb = MultinomialNB()
	mnb.fit(X_train, y_train)
	y_pred = mnb.predict(X_test)

	print('[MNB BoW] Accuracy: %f; F1-Score: %f' % (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))

	svm = LinearSVC(C=0.01)
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)

	print('[SVM BoW] Accuracy: %f; F1-Score: %f' % (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))