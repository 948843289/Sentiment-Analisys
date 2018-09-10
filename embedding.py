import os
import config
import numpy as np


def load_embedding(filename):
	# load embedding into memory
	with open(filename, 'r') as embedding_file:
		# create a map of words to vectors
		embedding = dict()
		dictionary = dict()
		for (i, line) in enumerate(embedding_file):
			values = line.split(' ')
			word = values[0]
			# key is string word, value is numpy array for vector
			embedding[word] = np.asarray(values[1:], dtype='float32')
			dictionary[word] = i + 1
	return embedding, dictionary


def get_embedding_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = np.zeros((vocab_size, config.EMBEDDING_DIM))
	# step vocab, store vectors using the Tokenizer's integer mapping
	"""for word in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector"""
	for i, word in enumerate(vocab):
		vector = embedding[word]
		weight_matrix[i] = vector
	return weight_matrix


# load embedding from file
print("Loading Embedding file...")
raw_embedding, dictionary = load_embedding("Embedding/glove/glove_custom_{}d_{}.txt".format(config.EMBEDDING_DIM, config.dataset_name))
# get vectors in the right order
print("Building Embedding Matrix...")
embedding_vectors = get_embedding_matrix(raw_embedding, dictionary.keys())

print("Saving file...")
embedding_matrix_filename = config.dataset_name + '_embedding_matrix_' + str(config.EMBEDDING_DIM) + '.npz'
word_embedding_filename = config.dataset_name + '_word_embedding_' + str(config.EMBEDDING_DIM) + '.npz'
dictionary_name = config.dataset_name + '_dictionary.npz'
np.savez_compressed(os.path.join(config.embedding_base_dir, embedding_matrix_filename), embedding_vectors)
np.savez_compressed(os.path.join(config.embedding_base_dir, word_embedding_filename), raw_embedding)
np.savez_compressed(os.path.join(config.embedding_base_dir, dictionary_name), dictionary)
print("Finished!!")
