import numpy as np
import keras
import h5py
import keras.backend as K
from keras import metrics
from time import time

from tensorboard_logging import Logger

class ValidCallBack(keras.callbacks.Callback):

	def __init__(self):
		super(ValidCallBack, self).__init__()
		
		# h5py file containing validation features and validation word embeddings
		self.F 			= h5py.File("./processed_features/validation_features.h5", "r")
		self.val_features= self.F["data/features"]

		wordF = h5py.File("./processed_features/embeddings.h5", 'r')
		self.word_embed	= wordF["data/word_embeddings"][:,:] 
		self.word_names  = map(lambda a:a[0], wordF["data/word_names"][:])
		wordF.close()

		self.image_fnames = map(lambda a:a[0], self.F["data/fnames"][:])
		
		print "[LOG] ValidCallBack: "
		print "val_feats: {} -- word_embed: {} -- word_names: {} -- image_fnames: {}".format(
				self.val_features.shape, self.word_embed.shape, len(self.word_names), len(self.image_fnames)
			) 
		
		# find all classes present in validation set 
		validation_classes = [cl.split("/")[-2] for cl in self.image_fnames]

		# Keep only those word_embed and word_names that are present in dataset 
		self.unique_classes = list(set(validation_classes))
		self.unique_classes_embed = []
		for cl in self.unique_classes:
			idx = self.word_names.index(cl)
			self.unique_classes_embed.append(self.word_embed[idx])
		self.unique_classes_embed = np.array(self.unique_classes_embed)
		self.unique_classes_embed = self.unique_classes_embed / np.linalg.norm(self.unique_classes_embed, axis=1, keepdims=True)

		self.mylogger = Logger("logs/top_{}".format(time()))


	def on_epoch_end(self, epoch, logs={}):

		accuracy_list = []
		for i in range(len(self.val_features)):

			trueclass 	= self.image_fnames[i].split("/")[-2]
			feat 		= self.val_features[i]

			preds = self.model.predict(feat.reshape((1,4096)))
			preds = preds / np.linalg.norm(preds)

			diff = self.unique_classes_embed - preds
			diff = np.linalg.norm(diff, axis=1)
			
			# min_idx = sorted(range(len(diff)), key=lambda x: diff[x])
			min_idx = np.argsort(diff)
			min_idx = min_idx[0:3]

			# print "current image of class {} | is closest to embedding of words:".format(trueclass)
			closest_words = []
			for i in min_idx:
				# print self.unique_classes[i]
				closest_words.append(self.unique_classes[i])

			# save closest words to accuracy_list 
			accuracy_list.append([self.image_fnames, trueclass, closest_words])
		
		# Display accuracy 
		top_1_acc = 0.0
		top_3_acc = 0.0
		for accuracy_data in accuracy_list:
			if accuracy_data[1] in accuracy_data[2][0:3]: # --- Top 1 Accuracy ---
				top_3_acc += 1
				if accuracy_data[1] == accuracy_data[2][0]: # --- Top 3 Accuracy ---
					top_1_acc += 1

		top_1_acc = round(top_1_acc/len(accuracy_list), 3)
		top_3_acc = round(top_3_acc/len(accuracy_list), 3)

		print "top 1: {} | top 3: {} ".format(top_1_acc, top_3_acc)

		print epoch
		self.mylogger.log_scalar("top1", float(top_1_acc), epoch)
		self.mylogger.log_scalar("top3", float(top_3_acc), epoch)
	
	def custom_for_keras(self, ALL_word_embeds):
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

class LoadValidationData():

	def __init__(self):

		# validation features (200x4096) and its true class (cat, dog, aeroplane.. 200)
		F 					= h5py.File("./processed_features/validation_features.h5", "r")
		self.val_features 	= F["data/features"][:]
		self.image_fnames 	= map(lambda a:a[0], F["data/fnames"][:])
		self.image_GT  		= [fname.split("/")[-2] for fname in self.image_fnames]
		F.close()

		# embeddings (400000x50) 
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













