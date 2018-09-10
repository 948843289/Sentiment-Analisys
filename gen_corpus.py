import os
import string
import config

from nltk import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')


def process_docs(directory, file):
    if config.dataset_name == "elec":
        # load the doc
        t = open(directory, 'r')
        for line in t:
            if line:
                line = line.replace('<br /><br />', ' ')
                table = str.maketrans('', '', string.punctuation)
                line = line.translate(table)
                line = line.strip()
                file.write(line + '\n')
        t.close()
    elif config.dataset_name == "imdb":
        # walk through all files in the folder
        for filename in os.listdir(directory):
            # create the full path of the file to open
            path = directory + filename
            # load the doc
            t = open(path, 'r')
            for line in t:
                if line:
                    line = line.replace('<br /><br />', ' ')
                    table = str.maketrans('', '', string.punctuation)
                    line = line.translate(table)
                    line = line.strip()
                    file.write(line + '\n')
            t.close()


# Open file to write in review
tmp_corpus = open("tmp_corpus_{}.txt".format(config.dataset_name), "w")

if config.dataset_name == "elec":
    process_docs(config.elec_train, tmp_corpus)
    process_docs(config.elec_unl, tmp_corpus)
elif config.dataset_name == "imdb":
    process_docs(config.imdb_train_positive, tmp_corpus)
    process_docs(config.imdb_train_negative, tmp_corpus)
    process_docs(config.imdb_train_unl, tmp_corpus)
else:
    print("Dataset name not set correctly (imdb or elec)!!")
    exit(0)

# Close file
tmp_corpus.close()

tmp_corpus = open("tmp_corpus_{}.txt".format(config.dataset_name), "r")
corpus = open("GloVe-1.2/corpus_{}".format(config.dataset_name), "w")

lines = tmp_corpus.readlines()
text = ''

# put everithing in one line
for line in lines:
    text = text + line
tmp_corpus.close()
os.remove("tmp_corpus_{}.txt".format(config.dataset_name))

# perform TOKENIZATION, returns a vector of words
words = tokenizer.tokenize(text.lower())

# rewrite this vector in a file in which words are separated by blank spaces
for w in words:
    corpus.seek(0, 2)
    corpus.write(w + ' ')
corpus.close()
