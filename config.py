import os


def load_doc(path):
	# Open the file
	file = open(path, 'r')
	text = file.read()
	# close the file
	file.close()
	return text


dataset_name = "imdb"  # "imdb" or "elec"
embedding_base_dir = "Embedding/"
if dataset_name == "elec":
	EMBEDDING_DIM = 512
	embedding_file = os.path.join(embedding_base_dir, "globe/glove_custom_512d.txt")
elif dataset_name == "imdb":
	EMBEDDING_DIM = 256
	embedding_file = os.path.join(embedding_base_dir, "globe/glove_custom_256d.txt")
else:
	print("Dataset name not set correctly (imdb or elec)!!")
	exit(0)

dataset_base_dir = "Dataset/"

imdb_dataset = os.path.join(dataset_base_dir, "Imdb/")
imdb_train = os.path.join(imdb_dataset, "train/")
imdb_test = os.path.join(imdb_dataset, "test/")
imdb_train_positive = os.path.join(imdb_train, "pos/")
imdb_train_negative = os.path.join(imdb_train, "neg/")
imdb_train_unl = os.path.join(imdb_train, "unsup/")
imdb_test_positive = os.path.join(imdb_test, "pos/")
imdb_test_negative = os.path.join(imdb_test, "neg/")

elec_unsupervised = True
if elec_unsupervised:
	elec_dataset = os.path.join(dataset_base_dir, "Elec/elec-unlab/")
	elec_train = os.path.join(elec_dataset, "elec-25k-train.txt")
	elec_unl = os.path.join(elec_dataset, "elec-25k-unlab00.txt")
	elec_test = os.path.join(elec_dataset, "elec-test.txt")
	elec_labels_train = os.path.join(elec_dataset, "elec-25k-train.cat")
	elec_labels_test = os.path.join(elec_dataset, "elec-test.cat")
else:
	elec_dataset = os.path.join(dataset_base_dir, "Elec/elec/")
	dim_elec = '25'  # 02 or 5 or 10 or 25 or 50 or 100 or 200
	if dim_elec == '02' or '5' or '10' or '25' or '50' or '100' or '200':
		elec_train = os.path.join(elec_dataset, "elec-{}k-train.txt".format(dim_elec))
		elec_test = os.path.join(elec_dataset, "elec-test.txt")
		elec_labels_train = os.path.join(elec_dataset, "elec-{}k-train.cat".format(dim_elec))
		elec_labels_test = os.path.join(elec_dataset, "elec-test.cat")
	else:
		print("Dimension of Elec dataset not set correctly!!")
		exit(0)
