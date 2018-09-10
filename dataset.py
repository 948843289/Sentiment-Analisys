import os
import config
import string
import numpy as np

from nltk import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')


def process_docs(directory, vocab):
	documents = list()
	if config.dataset_name == "imdb":
		# walk through all files in the folder
		for filename in os.listdir(directory):
			# create the full path of the file to open
			path = directory + filename
			# load the doc
			doc = config.load_doc(path)
			doc = doc.replace('<br /><br />', ' ')
			table = str.maketrans('', '', string.punctuation)
			doc = doc.translate(table)
			doc = doc.strip()
			tokens = tokenizer.tokenize(doc.lower())
			tokens = [(vocab[t] if (t in vocab) else vocab['<unk>']) for t in tokens]
			documents.append(tokens)
	elif config.dataset_name == "elec":
		# load the doc
		doc = open(directory, 'r')
		for line in doc:
			if line:
				line = line.replace('<br /><br />', ' ')
				table = str.maketrans('', '', string.punctuation)
				line = line.translate(table)
				line = line.strip()
				tokens = tokenizer.tokenize(line.lower())
				tokens = [(vocab[t] if (t in vocab) else vocab['<unk>']) for t in tokens]
				documents.append(tokens)
		doc.close()
	return documents


# load all training reviews
print("Loading and processing documents...")
if config.dataset_name == "imdb":
	# load the vocabulary
	vocab_name = config.dataset_name + '_dictionary.npz'
	vocab = np.load(os.path.join(config.embedding_base_dir, vocab_name))['arr_0'].item()

	train_positive_docs = process_docs(config.imdb_train_positive, vocab)
	train_negative_docs = process_docs(config.imdb_train_negative, vocab)
	train_docs = train_positive_docs + train_negative_docs
	X_train = np.asarray(train_docs)
	# define training labels
	y_train = np.array([1 for _ in range(len(train_positive_docs))] + [0 for _ in range(len(train_negative_docs))])

	# load unsup reviews
	train_unsup_doc = process_docs(config.imdb_train_unl, vocab)
	unlab = np.asarray(train_unsup_doc)

	# load all test reviews
	test_positive_docs = process_docs(config.imdb_test_positive, vocab)
	test_negative_docs = process_docs(config.imdb_test_negative, vocab)
	test_docs = test_positive_docs + test_negative_docs
	X_test = np.asarray(test_docs)
	# define testing labels
	y_test = np.array([1 for _ in range(len(test_positive_docs))] + [0 for _ in range(len(test_negative_docs))])
elif config.dataset_name == "elec":
	# load the vocabulary
	vocab_name = config.dataset_name + '_dictionary.npz'
	vocab = np.load(os.path.join(config.embedding_base_dir, vocab_name))['arr_0'].item()

	train_docs = process_docs(config.elec_train, vocab)
	X_train = np.asarray(train_docs)
	# Define training labels
	labels_train_file = open(config.elec_labels_train)
	text = labels_train_file.read()
	labels_train_file.close()
	labels_train = []
	for line in text:
		if line == '2' or line == '1':
			labels_train.append(int(line))
	y_train = np.asarray(labels_train)
	y_train[y_train == 1] = 0
	y_train[y_train == 2] = 1

	# load unsup reviews
	train_unsup_doc = process_docs(config.elec_unl, vocab)
	unlab = np.asarray(train_unsup_doc)

	# load all test reviews
	test_docs = process_docs(config.elec_test, vocab)
	X_test = np.asarray(test_docs)
	# define testing labels
	labels_test_file = open(config.elec_labels_test)
	text = labels_test_file.read()
	labels_test_file.close()
	labels_test = []
	for line in text:
		if line == '2' or line == '1':
			labels_test.append(int(line))
	y_test = np.asarray(labels_test)
	y_test[y_test == 1] = 0
	y_test[y_test == 2] = 1
else:
	print("Dataset name not set correctly (imdb or elec)!!")
	exit(0)


print("Saving file...")
np.savez_compressed(os.path.join(config.dataset_base_dir, 'X_train_{}.npz'.format(config.dataset_name)), X_train)
np.savez_compressed(os.path.join(config.dataset_base_dir, 'y_train_{}.npz'.format(config.dataset_name)), y_train)
np.savez_compressed(os.path.join(config.dataset_base_dir, "unlab_{}.npz".format(config.dataset_name)), unlab)
np.savez_compressed(os.path.join(config.dataset_base_dir, 'X_test_{}.npz'.format(config.dataset_name)), X_test)
np.savez_compressed(os.path.join(config.dataset_base_dir, 'y_test_{}.npz'.format(config.dataset_name)), y_test)

print("Finished!!")
