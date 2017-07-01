import numpy as np
import keras
import h5py

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

			print "current image of class {} | is closest to embedding of words:".format(trueclass)
			closest_words = []
			for i in min_idx:
				print self.unique_classes[i]
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
		print "top 1: {} | top 3: {} ".format(top_1_acc/len(accuracy_list), top_3_acc/len(accuracy_list))

			






