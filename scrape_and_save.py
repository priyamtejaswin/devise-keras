"""
Code to collect and parse captions for MSCOCO.
Images/captions are already separated into two sets for TRAIN and VAL.
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
from pycocotools.coco import COCO

print "\n\n\t\tCreating features.h5 and validation_features.h5 in processed_features\n\n"
tempF = h5py.File("processed_features/features.h5", "w")
tempF.create_group("data")
tempF.close()
tempF = h5py.File("processed_features/validation_features.h5", "w")
tempF.create_group("data")
tempF.close()
print "\n\n\t\tDONE\n\n"

WHITELIST = string.letters + string.digits
WORD_DIM = 300

COCO_ROOT = "/var/coco"

"""
loc_to_raw_file: location to json file for cap_type
cap_type: train/val/test
"""
loc_to_raw_file, cap_type = sys.argv[1], sys.argv[2]
assert os.path.isfile(loc_to_raw_file), "--File %s not found--"%(loc_to_raw_file)

# UNK_ix = glove_index["<unk>"]

print "\nLoading word_embeddings...\n"
emfp = h5py.File("processed_features/embeddings.h5", 'r') 
word_embeds	= emfp["data/word_embeddings"][:,:]
glove_index = {w[0]:word_embeds[n].tolist() for n,w in enumerate(emfp["data/word_names"][:])}
assert WORD_DIM==len(glove_index["the"]), "--Mismatch b/w WORD_DIM and glove data--"

_response = raw_input("\nParsing captions type %s at %s ...<y/n>?"%(cap_type, loc_to_raw_file))
if _response=='n':
	sys.exit()

cocoObj = COCO(loc_to_raw_file)

_c = 0
captions_list = []
image_TO_captions = defaultdict(list)

_cap_unk_words = 0
_unk_words = []
_cap_count = 0
caption_TO_count = {}

for capId, capInfo in cocoObj.anns.iteritems(): 
	assert capId==capInfo["id"], "Something is seriously wrong with the data!!!"

	caption_text = capInfo["caption"].strip().lower()

	tokens = []
	with_spaces = ""
	for char in caption_text:
		if char in WHITELIST:
			with_spaces+=char
		else:
			with_spaces+=" "

	if len(with_spaces.split())>=5: # Ultimate check for including a caption.
		new_list = []
		for word in with_spaces.split():
			if word in glove_index:
				new_list.append(word)
			else:
				## current strategy is to skip unknown words!
				# new_list.append("<unk>")
				print "UNK WORD %s:"%(word), with_spaces
				_unk_words.append(word)
				_c+=1
		if _c>0:
			_cap_unk_words+=1
		else:
			## Only associate with the image if NO unk WORDS found.
			image_TO_captions[capInfo["image_id"]].append(capId)
			captions_list.append(" ".join(new_list).encode("utf-8"))
			caption_TO_count[capId] = _cap_count
			_cap_count+=1

		_c=0

	else:
		print "LEN <5:", caption_text

print "Unk words", _unk_words
print "Len unkown words", len(_unk_words), len(set(_unk_words))
print "Captions with unk words", _cap_unk_words


#ipdb.set_trace()

_response = raw_input("\nFinished parsing %s. Proceed with saving processed text data and meta-data?<y/n>"%(loc_to_raw_file))
if _response=='n':
	sys.exit()

_counts = [len(c.strip().split()) for c in captions_list]
print sorted(Counter(_counts))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#ipdb.set_trace()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions_list)
sequences = tokenizer.texts_to_sequences(captions_list)

word_index = tokenizer.word_index
print('\nFound %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

print "\nCreating embedding_layer weights for type %s..."%(cap_type)
embedding_layer = np.zeros((len(word_index) + 1, WORD_DIM))
for word,ix in word_index.items():
		embedding_layer[ix, :] = glove_index[word]

print "\nSaving embedding and word_index to disk...\n"

print "\t\tembedding matrix CONTAINS <pad> at the 0 index. Size:", embedding_layer.shape
pickle.dump(embedding_layer, open("KERAS_embedding_layer.pkl", "w"))

print "\t\tword index DOES NOT CONTAIN <pad>. The index starts from 0. Size:", len(word_index)
pickle.dump(word_index, open("DICT_word_index.pkl", "w"))

print "\t\tcaption_data contains ALL the captions."
pickle.dump(data, open("ARRAY_caption_data.pkl", "w"))

print "\t\timage_TO_captions contains the CAPTION_IDS for every IMAGE_ID."
pickle.dump(image_TO_captions, open("DICT_image_TO_captions.pkl", "w"))

image_TO_tokens = {}
for imgId, list_of_caption_ids in image_TO_captions.iteritems():
	image_TO_tokens[imgId] = [data[caption_TO_count[cid]] for cid in list_of_caption_ids]
print "\t\timage_TO_tokens contains the list of TOKEN_IDS for every IMAGE_ID."
pickle.dump(image_TO_tokens, open("DICT_image_TO_tokens.pkl", "w"))

print "\nDONE.\n"
