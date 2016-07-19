__author__ = 'thomas'
from argparse import ArgumentParser
import logging
import os
import sys

from apt_toolkit.utils.base import path_utils
from apt_toolkit.utils import vector_utils
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from scipy import sparse
from sklearn import cross_validation
import numpy as np
import scipy


parser = ArgumentParser()
parser.add_argument('-a', '--action', type=str, required=True, help='action to be executed')
parser.add_argument('-ip', '--input-path', type=str, help='path to input vectors')
parser.add_argument('-i', '--input-file', type=str, help='input file')
parser.add_argument('-op', '--output-path', type=str, help='path to output')
parser.add_argument('-o', '--output-file', type=str, help='output file')
parser.add_argument('-pf', '--phrases-file', type=str, help='phrases file')


def autoencoder_apt(vector_file, output_file, phrases_file):
	logging.info('Loading vector file={}...'.format(vector_file))
	vectors = vector_utils.load_vector_cache(vector_file)
	logging.info('Vector file loaded!')

	# Convert APT cache to sparse matrix
	logging.info('Converting APT cache to sparse matrix...')
	row_names, X = vector_utils.dict_to_sparse(vectors)
	logging.info('Converted!')

	P = sparse.lil_matrix(np.zeros((216, X.shape[1] * 2)))

	logging.info('Collecting phrase pairs...')
	phrases = []
	with open(phrases_file, 'r') as pf:
		for idx, line in enumerate(pf):
			w1, w2 = line.strip().split()

			v1 = X[row_names.index(w1)] if w1 in row_names else sparse.lil_matrix(np.zeros((X.shape[1],)))
			v2 = X[row_names.index(w2)] if w2 in row_names else sparse.lil_matrix(np.zeros((X.shape[1],)))

			phrases.append('_'.join([w1, w2]))

			P[idx] = sparse.hstack((v1, v2), format='lil')
	logging.info('Phrase pairs collected!')

	# Train/Dev Split
	X_train, X_test, train_phrases, test_phrases = cross_validation.train_test_split(P.tocsr(), np.array(phrases), test_size=0.2)

	# Need dense arrays because keras sucks for sparse shit
	X_train = X_train.toarray()
	X_test = X_test.toarray()
	train_phrases = train_phrases.tolist()
	test_phrases = test_phrases.tolist()

	input_dim = P.shape[1]
	encoding_dim = 300

	input_vector = Input(shape=(input_dim,))
	encoded = Dense(encoding_dim, activation='tanh')(input_vector)
	decoded = Dense(input_dim, activation='tanh')(encoded)

	autoencoder = Model(input=input_vector, output=decoded)

	encoder = Model(input=input_vector, output=encoded)
	encoded_input = Input(shape=(encoding_dim,))

	decoder_layer = autoencoder.layers[-1]
	decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

	autoencoder.compile(optimizer='adadelta', loss='cosine_proximity')

	autoencoder.fit(X_train, X_train,
					nb_epoch=5,
					batch_size=16,
					shuffle=True,
					validation_data=(X_test, X_test))

	E = encoder.predict(X_test)

	print(E.shape)
	print(test_phrases)


def autoencoder_test():
	# this is the size of our encoded representations
	encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

	# this is our input placeholder
	input_img = Input(shape=(784,))
	# "encoded" is the encoded representation of the input
	encoded = Dense(encoding_dim, activation='relu')(input_img)
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(784, activation='sigmoid')(encoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input=input_img, output=decoded)

	# this model maps an input to its encoded representation
	encoder = Model(input=input_img, output=encoded)

	# create a placeholder for an encoded (32-dimensional) input
	encoded_input = Input(shape=(encoding_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-1]
	# create the decoder model
	decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	(x_train, _), (x_test, _) = mnist.load_data()

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

	print(x_train.shape)
	print(x_test.shape)

	autoencoder.fit(x_train, x_train,
					nb_epoch=50,
					batch_size=256,
					shuffle=True,
					validation_data=(x_test, x_test))

if (__name__ == '__main__'):
	args = parser.parse_args()

	timestamped_foldername = path_utils.timestamped_foldername()
	log_path = os.path.join(path_utils.get_log_path(), timestamped_foldername)
	if (not os.path.exists(log_path)):
		os.makedirs(log_path)

	log_formatter = logging.Formatter(fmt='%(asctime)s: %(levelname)s - %(message)s', datefmt='[%d/%m/%Y %H:%M:%S %p]')
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.DEBUG)

	file_handler = logging.FileHandler(os.path.join(log_path, 'wikipedia_clean_preprocess_{}.log'.format(args.action)))
	file_handler.setFormatter(log_formatter)
	root_logger.addHandler(file_handler)

	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(log_formatter)
	root_logger.addHandler(console_handler)

	if (not os.path.exists(args.output_path)):
		os.makedirs(args.output_path)

	if (args.action == 'autoencoder_test'):
		autoencoder_test()
	elif (args.action == 'autoencoder_apt'):
		autoencoder_apt(vector_file=os.path.join(args.input_path, args.input_file),
						output_file=os.path.join(args.output_path, args.output_file),
						phrases_file=args.phrases_file)