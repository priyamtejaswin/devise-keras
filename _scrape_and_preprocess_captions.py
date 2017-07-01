"""
Creates two folders: _data & _validation.
Validation contains every nth image from the main set.
"""

import os
import re
import sys
import h5py
import ipdb
from collections import defaultdict, Counter
import string
import pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil

from extract_features_and_dump import dump_wv_to_h5

file_path, embeddings_path = sys.argv[1], sys.argv[2]

image_re = re.compile('<td><img src="(.*)\/(.*)"><\/td>')
caption_re = re.compile('<tr><td>(.*)<\/td><\/tr>')

VAL_NUM = 10
UIUC_ROOT = "UIUC_PASCAL_DATA"
UIUC_VAL = "UIUC_PASCAL_VAL"
UIUC_URL = "http://vision.cs.uiuc.edu/pascal-sentences"
WHITELIST = string.letters + string.digits
WORD_DIM = 50

print "Loading glove vector data..."
F = h5py.File("processed_features/features.h5", "w")
data = F.create_group("data")
dt   = h5py.special_dtype(vlen=str)
# extract and dump word vectors
print "Dumping word embeddings..."
_ = data.create_dataset("word_embeddings", (0, WORD_DIM), maxshape=(None, WORD_DIM))
_ = data.create_dataset("word_names", (0, 1), dtype=dt, maxshape=(None, 1))

glove_index = {}
f = open(embeddings_path)
word_batch, vector_batch, _c = [], [], 0

for line in f.readlines():
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_index[word] = coefs
    word_batch.append(word)
    vector_batch.append(coefs)
    
    if _c%1000==0 and _c>0:
    	dump_wv_to_h5(word_batch, vector_batch, F)
    	word_batch, vector_batch = [], []

    _c+=1

dump_wv_to_h5(word_batch, vector_batch, F) # to catch the trailing vectors
f.close()
F.close()
print 'Found %s word vectors.' % len(glove_index)

# Copying features.h5 to validation_features.h5
shutil.copy2("processed_features/features.h5", "processed_features/validation_features.h5")
print "Copied features.h5 to validation_features.h5"

answer = raw_input("DO you want to continue with downloading UIUC_PASCAL stuff?")
if answer == "n":
	sys.exit(0)

# UNK_ix = glove_index["<unk>"]

if os.path.exists(UIUC_ROOT):
	print "\nUIUC_PASCAL_DATA detected. The program has stopped here. Press ENTER to continue downloading all data. Press CTRL+C to exit program now.\n"
else:
	os.makedirs(UIUC_ROOT)
if os.path.exists(UIUC_VAL):
	print "\nUIUC_PASCAL_VAL detected. The program has stopped here. Press ENTER to continue downloading all data. Press CTRL+C to exit program now.\n"
else:
	os.makedirs(UIUC_VAL)

print "\nParsing and downloading html source...\n"

_c = 0
captions_list = []
id_TO_class = {}
class_TO_images = defaultdict(list)
image_TO_captions = defaultdict(list)

image_count = -1
caption_count = -1
uniq_class = set()

with open(file_path, 'r') as fp:
	for line in fp.readlines():
		clean = line.strip()

		match_image = image_re.search(clean)
		if match_image:
			image_count+=1

			class_name = match_image.group(1)
			image_name = match_image.group(2)

			uniq_class.add(class_name)



			dir_name = os.path.join(UIUC_ROOT, class_name)
			if not os.path.exists(dir_name):
				os.makedirs(dir_name)

			img_name = os.path.join(dir_name, image_name)
			img_url = os.path.join(UIUC_URL, class_name, image_name)
			system_string = "wget %s -O %s"%(img_url, img_name)

			if not os.path.exists(img_name):
				os.system(system_string)

			id_TO_class[len(uniq_class) -1] = class_name
			class_TO_images[len(uniq_class) -1].append(image_count)
			
			continue

		match_caption = caption_re.search(clean)
		if match_caption:
			_c+=1
			
			caption_text = match_caption.group(1).strip().lower()

			tokens = []
			with_spaces = ""
			for char in caption_text:
				if char in WHITELIST:
					with_spaces+=char
				else:
					with_spaces+=" "+char

			if with_spaces[-1] in ".!":
				with_spaces = with_spaces[:-1]

			if len(with_spaces.split())>=5: # Ultimate check for including a caption.
				caption_count+=1
				image_TO_captions[image_count].append(caption_count)

				new_list = []
				for word in with_spaces.split():
					if word in glove_index:
						new_list.append(word)
					else:
						## current strategy is to skip unknown words!
						# new_list.append("<unk>")
						pass

				captions_list.append(" ".join(new_list))

				print caption_text

		print image_count, caption_count
		if _c==3*50*5:
			print "Downloaded and processed %d images, %d captions"%(image_count+1, caption_count+1)
			_response = raw_input("Download more?<y/n>:")
			if _response=="n":
				break
			

print "\nFinished downloading all images. Proceed with saving processed text data and meta-data?\nCtrl-C to exit.\n"
raw_input()

_counts = [len(c.strip().split()) for c in captions_list]
print Counter(_counts)

# plt.hist(_counts, 50)
# plt.show()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions_list)
sequences = tokenizer.texts_to_sequences(captions_list)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

# missing_indices = [v for k,v in word_index.items() if k not in glove_index]
# for ix in missing_indices:
	# data[data==ix] = UNK_ix
# print "Missing indices:", missing_indices
# print "Replaced missing with <unk>"

print "Creating embedding_layer weights..."
embedding_layer = np.zeros((len(word_index) + 1, 50))
for word,ix in word_index.items():
		embedding_layer[ix, :] = glove_index[word]

print "Saving embedding and word_index to disk..."
pickle.dump(embedding_layer, open("KERAS_embedding_layer.pkl", "w"))
pickle.dump(word_index, open("DICT_word_index.pkl", "w"))
pickle.dump(data, open("ARRAY_caption_data.pkl", "w"))

pickle.dump(id_TO_class, open("DICT_id_TO_class.pkl", "w"))
pickle.dump(class_TO_images, open("DICT_class_TO_images.pkl", "w"))
pickle.dump(image_TO_captions, open("DICT_image_TO_captions.pkl", "w"))

print "DONE"
# ipdb.set_trace()
