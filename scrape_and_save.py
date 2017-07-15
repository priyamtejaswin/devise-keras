"""
Code to scrape images from UIUC and parse the captions.
Creates two folders: _data & _validation.
Validation contains every nth image and mth caption from the main set.
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
import urllib

print "\n\n\t\tCreating features.h5 and validation_features.h5 in processed_features\n\n"
tempF = h5py.File("processed_features/features.h5", "w")
tempF.create_group("data")
tempF.close()
tempF = h5py.File("processed_features/validation_features.h5", "w")
tempF.create_group("data")
tempF.close()
print "\n\n\t\tDONE\n\n"

file_path = sys.argv[1]
ENV = sys.argv[2]

image_re = re.compile('<td><img src="(.*)\/(.*)"><\/td>')
caption_re = re.compile('<tr><td>(.*)<\/td><\/tr>')

UIUC_ROOT = "UIUC_PASCAL_DATA"
UIUC_VAL = "UIUC_PASCAL_VAL"
UIUC_URL = "http://vision.cs.uiuc.edu/pascal-sentences"
WHITELIST = string.letters + string.digits
WORD_DIM = 300

answer = raw_input("DO you want to continue with downloading UIUC_PASCAL stuff?<y/n>")
if answer == "n":
	sys.exit(0)

# UNK_ix = glove_index["<unk>"]

if os.path.exists(UIUC_ROOT):
	raw_input("\nUIUC_PASCAL_DATA detected. The program has stopped here. Press ENTER to continue downloading all data. Press CTRL+C to exit program now.\n")
else:
	os.makedirs(UIUC_ROOT)
if os.path.exists(UIUC_VAL):
	raw_input("\nUIUC_PASCAL_VAL detected. The program has stopped here. Press ENTER to continue downloading all data. Press CTRL+C to exit program now.\n")
else:
	os.makedirs(UIUC_VAL)

print "\nLoading word_embeddings...\n"
emfp = h5py.File("processed_features/embeddings.h5", 'r') 
word_embeds	= emfp["data/word_embeddings"][:,:]
glove_index = {w[0]:word_embeds[n].tolist() for n,w in enumerate(emfp["data/word_names"][:])}

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
			class_name_orig = class_name
			## Some hardcoding here.
			if class_name=="diningtable":
				class_name = "table"
			if class_name=="pottedplant":
				class_name = "plant"
			if class_name=="tvmonitor":
				class_name = "tv"

			image_name = match_image.group(2)

			uniq_class.add(class_name)

			dir_name = os.path.join(UIUC_ROOT, class_name)
			if not os.path.exists(dir_name):
				os.makedirs(dir_name)

			img_name = os.path.join(dir_name, image_name)
			img_url = os.path.join(UIUC_URL, class_name_orig, image_name)
			system_string = "wget %s -O %s"%(img_url, img_name)

			if not os.path.exists(img_name):
				#os.system(system_string)
				urllib.urlretrieve(img_url, img_name)

			id_TO_class[len(uniq_class) -1] = class_name
			class_TO_images[len(uniq_class) -1].append(image_count)
			
			continue

		match_caption = caption_re.search(clean)
		if match_caption:
			_c+=1 # tracking number of captions which have been covered

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
		if ENV != "PROD":
			if (_c%(3*50*5)==0) and (image_count>0):
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
print('\nFound %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

# missing_indices = [v for k,v in word_index.items() if k not in glove_index]
# for ix in missing_indices:
	# data[data==ix] = UNK_ix
# print "Missing indices:", missing_indices
# print "Replaced missing with <unk>"

print "\nCreating embedding_layer weights..."
embedding_layer = np.zeros((len(word_index) + 1, WORD_DIM))
for word,ix in word_index.items():
		embedding_layer[ix, :] = glove_index[word]

print "\nSaving embedding and word_index to disk..."

print "\t\tembedding matrix CONTAINS <pad> at the 0 index. Size:", embedding_layer.shape
pickle.dump(embedding_layer, open("KERAS_embedding_layer.pkl", "w"))

print "\t\tword index DOES NOT CONTAIN <pad>. The index starts from 0. Size:", len(word_index)
pickle.dump(word_index, open("DICT_word_index.pkl", "w"))

print "\t\tcaption_data contains ALL the captions."
print "Use id_TO_class, class_TO_images, image_TO_captions for collecting the paired captions."
pickle.dump(data, open("ARRAY_caption_data.pkl", "w"))

pickle.dump(id_TO_class, open("DICT_id_TO_class.pkl", "w"))
pickle.dump(class_TO_images, open("DICT_class_TO_images.pkl", "w"))
pickle.dump(image_TO_captions, open("DICT_image_TO_captions.pkl", "w"))

print "\nDONE\nRemember that the class_TO_images dict has to be udpated after shuffling validation data."
