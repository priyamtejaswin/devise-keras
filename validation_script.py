import numpy as np
import keras
import h5py

class ValidCallBack(keras.callbacks.Callback):

	def __init__(self, folder):
		super(ValidCallBack, self).__init__()
		
		# h5py file containing validation features and validation word embeddings
		F 			= h5py.File("./processed_features/validation.h5", "r")
		val_features= F["data/features"]
		val_embed	= F["data/word_embeddings"][:,:] 
		word_names  = F["data/word_names"]
		image_fnames = map(lambda a:a[0], F["data/fnames"][:])

		print "val_feats: {} -- val_embed: {} -- word_names: {} -- image_fnames: {}".format(
				val_features.shape, val_embed.shape, word_names.shape, image_fnames.shape
			) 

		print "..loading up validation image features and its true class into memory"
		self.all_features = []
		self.all_trueclass = []
		for i in range(len(image_fnames)):

			self.all_features.append(val_features[i])
			self.all_trueclass.append(image_fnames[i].split("/")[-2])

		print "..loaded ^"

		assert len(self.all_features) == len(self.all_trueclass), "Length of features == length of trueclass"

	def on_epoch_end(self, epoch, logs={}):

		for feat, trueclass in zip(self.all_features, self.all_trueclass):

			preds = self.model.predict(feat)
			preds = preds / np.linalg.norm(preds)

			diff = val_embed - preds
			diff = np.linalg.norm(diff, axis=1)

			min_idx = sorted(range(len(diff)), key=lambda x: diff[x])
			min_idx = min_idx[0:3]
			# now min index contains the indexes of embeddings that 
			# are close the predicted embeddings 
			# now just display names of these min_idx embeddings 

			print "Closest classes to {} are :".format(trueclass)
			for i in min_idx:
				print word_names[i][0]  

			






