import numpy as np
import ipdb
import keras
import h5py
import keras.backend as K
from keras import metrics
from time import time
import cPickle as pickle
from tensorboard_logging import Logger
from itertools import izip
from tqdm import *

class ValidCallBack(keras.callbacks.Callback):

	def __init__(self,
		PATH_image_to_captions="DICT_image_TO_tokens.VAL.pkl",
		PATH_image_features="processed_features/validation_features.h5",
		PATH_word_index="DICT_word_index.pkl"
		):

		super(ValidCallBack, self).__init__()
		
		# h5py file containing validation features and validation word embeddings
		print "I am here.."
		self.F 			  = h5py.File(PATH_image_features, "r")
		self.val_features = self.F["data/features"] ## ALERT - DO NOT LOAD EVERYTHING!
		self.len_img_feats = self.val_features.shape[0]

		# load word indices 
		self.word_index = pickle.load(open(PATH_word_index))
		
		print "[LOG] ValidCallBack: "
		print "val_feats: {}".format(self.val_features.shape) 
		
		# ipdb.set_trace()
		# Load ALL caption data 
		image_to_captions = pickle.load(open(PATH_image_to_captions))

		# filter out the validation captions from "all_captions"
		self.val_to_caption = []
		for imgId, capList in image_to_captions.iteritems():
			for cap in capList:
				self.val_to_caption.append((imgId, cap))

		# EXPECTATION
		# now we expect self.val_to_caption to be a list of 
		# (string caption, val_features index) tuples.
		self.len_cap_feats = len(self.val_to_caption)
		self.mylogger = Logger("logs/top_{}".format(time()))
		# ipdb.set_trace()

	def on_epoch_end(self, epoch, logs={}):
		BATCH_SIZE = 500 ## batch size for running forward passes.

		# running forward pass for image_feats + dummy captions
		MAX_SEQUENCE_LENGTH = 20
		WORD_DIM = 300
		# preds = self.model.predict([self.val_features, np.zeros((len(self.val_features), 20))]) 
		# im_outs = preds[:, :WORD_DIM]

		_img_ix_gen = zip(
			range(0, self.len_img_feats, BATCH_SIZE), 
			range(BATCH_SIZE, self.len_img_feats, BATCH_SIZE)
		)
		if _img_ix_gen[-1][1]!=self.len_img_feats:
			_img_ix_gen.append((_img_ix_gen[-1][1], self.len_img_feats))

		preds = [
			self.model.predict([self.val_features[lx:ux, :], np.zeros((ux-lx, MAX_SEQUENCE_LENGTH))])[:, :WORD_DIM]
			for lx,ux in _img_ix_gen
		]
		im_outs = np.concatenate(preds, axis=0)

		# runnign forward pass for dummy feats + actual captions 
		
		_cap_ix_gen = zip(
			range(0, self.len_cap_feats, BATCH_SIZE),
			range(BATCH_SIZE, self.len_cap_feats, BATCH_SIZE)
		)
		if _cap_ix_gen[-1][1]!=self.len_cap_feats:
			_cap_ix_gen.append((_cap_ix_gen[-1][1], self.len_cap_feats))

		just_captions = [cap for _,cap in self.val_to_caption]
		just_indices  = [val_idx for val_idx,_ in self.val_to_caption]

		# preds = self.model.predict([ np.zeros((len(just_captions),4096)), just_captions])
		# cap_out = preds[:, WORD_DIM:]

		#ipdb.set_trace()

		preds = [
			self.model.predict([np.zeros((ux-lx, 4096)), np.array(just_captions[lx:ux])])[:, WORD_DIM:]
			for lx,ux in _cap_ix_gen
		]
		cap_out = np.concatenate(preds, axis=0)

		#ipdb.set_trace()

		# normalize the outputs
		im_outs = im_outs / np.linalg.norm(im_outs, axis=1, keepdims=True)
		cap_out = cap_out / np.linalg.norm(cap_out, axis=1, keepdims=True)

		TOP_K = 5
		correct = 0.0	
		for i in tqdm(xrange(len(cap_out))):

			diff = im_outs - cap_out[i] 
			diff = np.linalg.norm(diff, axis=1)
			top_k_indices = np.argsort(diff)[:TOP_K].tolist()

			correct_index = just_indices[i] 
			if correct_index in top_k_indices:
				correct += 1.0
		print "validation accuracy: ", correct / len(cap_out)
		print "num correct : ", correct
		self.mylogger.log_scalar(tag="top_5", value= correct / len(cap_out) , step = epoch)

		# # REPEAT cap_out 
		# cap_out_repeated = np.repeat( cap_out, repeats=len(im_outs), axis=0 )

		# # TILE im_outs 
		# im_outs_tile     = np.tile( im_outs, reps=(len(cap_out),1) )

		# # just a check
		# assert im_outs_tile.shape[0] == cap_out_repeated.shape[0], "tiled and repeated matrices MUST have same num of rows"

		# # do comparison 
		# diff = im_outs_tile - cap_out_repeated
		# diff = np.linalg.norm(diff, axis=1)

		# TOP_K = 5
		# correct = 0.0
		# for i in range(len(cap_out)):

		# 	diff_for_that_caption = diff[ i*len(im_outs) : i*len(im_outs) + len(im_outs) ]
		# 	top_k_indices = np.argsort(diff_for_that_caption)[:TOP_K].tolist()

		# 	correct_index = just_indices[i]
		# 	if correct_index in top_k_indices:
		# 		correct += 1
		
		# # calculate TOP_K accuracy 
		# accuracy_top_k = correct / len(self.val_to_caption)
		# print "Validation accuracy: ",accuracy_top_k
		# self.mylogger.log_scalar(tag="top_5", value=accuracy_top_k, step=epoch)

	
	def custom_for_keras(self, ALL_word_embeds): # DEPRECATED #
	
		## only the top 20 rows from word_vectors is legit!
		def top_accuracy(true_word_indices, image_vectors):
			l2 = lambda x, axis: K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
			l2norm = lambda x, axis: x/l2(x, axis)

			l2_words = l2norm(ALL_word_embeds, axis=1)
			l2_images = l2norm(image_vectors, axis=1)

			tiled_words = K.tile(K.expand_dims(l2_words, axis=1) , (1, 200, 1))
			tiled_images = K.tile(K.expand_dims(l2_images, axis=1), (1, 20, 1))

			diff = K.squeeze(l2(l2_words - l2_images, axis=2))

			# slice_top3 = lambda x: x[:, 0:3]
			# slice_top1 = lambda x: x[:, 0:1]

			diff_top5 = metrics.top_k_categorical_accuracy(tiled_images, diff)
			return diff_top5
			
		return top_accuracy

class LoadValidationData(): # DEPRECATED #

	def __init__(self):
		# DEPRECATED #
		# validation features (200x4096) and its true class (cat, dog, aeroplane.. 200)
		F 					= h5py.File("./processed_features/validation_features.h5", "r")
		self.val_features 	= F["data/features"][:]
		self.image_fnames 	= map(lambda a:a[0], F["data/fnames"][:])
		self.image_GT  		= [fname.split("/")[-2] for fname in self.image_fnames]
		F.close()

		# embeddings (400000xWORD_DIM) 
		wordF = h5py.File("./processed_features/embeddings.h5", 'r')
		self.word_embed	= wordF["data/word_embeddings"][:,:] 
		self.word_names  = map(lambda a:a[0], wordF["data/word_names"][:])
		wordF.close()

		# unique classes in validation set and their embeddings 
		self.unique_classes = list(set(self.image_GT))
		self.unique_classes_embed = []
		for cl in self.unique_classes:
			idx = self.word_names.index(cl)
			self.unique_classes_embed.append(self.word_embed[idx])
		self.unique_classes_embed = np.array(self.unique_classes_embed)
		self.unique_classes_embed = self.unique_classes_embed / np.linalg.norm(self.unique_classes_embed, axis=1, keepdims=True)

		# convert self.image_GT ==> self.image_GT_indices based on self.unique_classes
		self.image_GT_indices = [] 
		for cl in self.image_GT:
			self.image_GT_indices.append(self.unique_classes.index(cl))

		assert len(self.image_GT_indices) 	== len(self.val_features)
		assert len(self.image_GT) 			== len(self.val_features)

	def get_data(self):
		return self.val_features, np.array(self.image_GT_indices)[:, np.newaxis]












