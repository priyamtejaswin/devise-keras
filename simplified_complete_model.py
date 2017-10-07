import numpy as np
import pickle 
from extract_features_and_dump import define_model
from rnn_model import hinge_rank_loss
import keras 
from keras.models import Model, load_model 
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Embedding, LSTM, concatenate
import keras.backend as K
from keras.layers import Lambda
import sys, os 
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from extract_features_and_dump import define_model as TheVGGModel
from lime import lime_image
from scipy.spatial.distance import cdist
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import urllib
import cStringIO
import datetime

class FullModel(object):
	"""
	Unified vision(VGG) and language(RNN) model.
	Accepts paths to rnn_model, vgg_weights and word_index.
	See def:TEST_lime for usage.
	"""
	
	def __init__(self, rnn_model_path, word_index_path, vgg_weights_path):
		print "\n\tCreating model...loading weights and parameters...\n"

		# some globals 
		self.WORD_DIM = 300
		self.BATCH = 1 
		
		assert os.path.isfile(rnn_model_path), "Provided incorect location to rnn_model"
		assert os.path.isfile(word_index_path), "Provided incorrect location to word_index_path"
		assert os.path.isfile(vgg_weights_path), "Provided incorrect location to vgg_weights"

		with open(word_index_path, 'r') as fp:
			self.word_index = pickle.load(fp)
			assert isinstance(self.word_index, dict), "word_index_path provided is not a dict"

		# vgg16 - fc2 output
		# top_model = VGG16(weights="imagenet", include_top="True")
		# self.top_model = Model(inputs=top_model.input, outputs=top_model.get_layer("fc2").output)
		self.top_model = TheVGGModel(vgg_weights_path)

		# rnn part + recompile with new loss
		self.rnn_model = load_model(rnn_model_path, custom_objects={"hinge_rank_loss":hinge_rank_loss})
		self.rnn_model.compile(optimizer="rmsprop", loss=self.custom_distance_function)

		print "\n\tDONE\n"

	def custom_distance_function(self, y_true, y_pred, DEBUG=False):
		select_images = lambda x: x[:, :self.WORD_DIM]
		select_words = lambda x: x[:, self.WORD_DIM:]

		# separate the images from the captions==words
		image_vectors = Lambda(select_images, output_shape=(self.BATCH, self.WORD_DIM))(y_pred)
		word_vectors = Lambda(select_words, output_shape=(self.BATCH, self.WORD_DIM))(y_pred)

		# l2 norm
		l2 = lambda x: K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
		l2norm = lambda x: x/l2(x)
		
		image_vectors = l2norm(image_vectors)
		word_vectors = l2norm(word_vectors)

		# correct_image VS incorrect_words | Note the singular/plurals
		print "\n\n\t\tAM I PRINTING/COMPILING THE NEW CUSTOM LOSS??\n\n"
		cost = K.sum(image_vectors * word_vectors, axis=1)
		# cost_images = K.sum(cost_images, axis=-1) ## Commented because values expected for multiple inputs.

		if DEBUG:
			import ipdb
			ipdb.set_trace()

		return cost ## WAS EARLIER DIVIDING BY INCORRECT_self.BATCH WHICH HAS BEEN SET TO 0, IDIOT.

	def predict(self, x):
		"""
		Wrapper exposed to LIME.
		The x is only the image. The caption is at self.caption .
		Runs the end-to-end FullModel and returns the cosine 
		distance b/w the image(s) and the caption.
		"""
		assert len(x.shape)==4, "--WHOA, the shape is not 4?????--"
		# import ipdb; ipdb.set_trace()
		assert x.min()==0
		assert x.max()<=1

		if x.shape[-1]==3:
			print "CHANGING SHAPE"
			x_rolled = np.swapaxes(np.swapaxes(x, 3, 2), 2, 1)
		else:
			print "NOT CHANGING SHAPE"
			x_rolled = x

		print x_rolled.shape

		## pass through vgg 
		top_op = self.top_model.predict(
			preprocess_input(x_rolled * 255.0) ## mean-subtraction DONE HERE
		)
		# import ipdb;ipdb.set_trace()

		## pass through rnn model (get loss) 
		rnn_output = self.rnn_model.predict( 
			[top_op, np.tile(self.caption, (x_rolled.shape[0], 1))]
		) ## returns a (n, 600) array

		print rnn_output.shape

		## calculate the distance b/w image perturbations and the caption
		word_vectors = rnn_output[:, :self.WORD_DIM]
		caption_vectors = rnn_output[:, self.WORD_DIM:]
		loss = 1 - cdist(word_vectors, caption_vectors, metric="cosine")

		return loss

	@staticmethod
	def preprocess_image(img_path):
		"""
		!!!!The mean-subtraction is done JUST before the VGG pass!!!! 
		Returns by expanding dims for "batch" dimension
		and type casting to np.float64: just scikit-image things...
		"""
		img_input = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img_input)
		x = np.expand_dims(x, axis=0)
		x = x.astype(np.float64)/255.0
		x = np.swapaxes(np.swapaxes(x, 1, 2), 2, 3) ## Moving to channels last.
		x = x[0] ## Eliminate 'batch' dimension.
		return x

	def preprocess_caption(self, caption_string):
		"""
		Accepts a string - byte or uni.
		Assumes its already cleaned and everything from the JS ui.
		Splits into words and maps to the word_index.
		Returns the string and an array of indices with 0 padding.
		"""
		if isinstance(caption_string, unicode):
			str_caption = caption_string.encode('utf-8')
		else:
			str_caption = caption_string

		split_caption = str_caption.split(' ')
		if len(split_caption)>20:
			print "---WARNING: Input string contains more than 20 words. TRUNCATING.---"
			split_caption = split_caption[:20] ## Max-len 20.

		array_caption = []
		for word in split_caption:
			assert word in self.word_index, "---ERROR: word '%s' not in self.word_index---"%(word)
			array_caption.append(self.word_index[word])

		array_caption = array_caption + [0 for _ix in range(20 - len(array_caption))] ## Max-len 20.
		return str_caption, array_caption

	def run_lime(self, image_url, caption_string):
		"""
		Method to be called from the backend.
		Accepts an image_url and the caption_string.
		Runs LIME. Returns the mask.
		"""
		lime_BATCH = 500 

		try:
			imgFile = cStringIO.StringIO(urllib.urlopen(image_url).read()) ## Download image.
		except Exception as error:
			print "--LOG--%s"%(str(datetime.datetime.now())), error
			return np.zeros((224, 224, 3))
		
		xImg = self.preprocess_image(imgFile) ## Load, pre-process image.

		self.str_caption, self.caption = self.preprocess_caption(caption_string) ## Pre-process caption.

		print "\n\t\tRunning LIME\n"
		_st = time.time()

		explainer = lime_image.LimeImageExplainer() ## LIME explainer.
		explanation = explainer.explain_instance(
			xImg,
			self.predict,
			top_labels=1, hide_color=0, batch_size=lime_BATCH, num_samples=5000, num_features=100
		)

		print "\n\t\tDONE. Took", (time.time() - _st)/60.0, "minutes.\n"
		
		tmpImg, tmpMask = explanation.get_image_and_mask(
			label=lime_BATCH-1, positive_only=True, num_features=10, hide_rest=True
		)
		return tmpMask
	
def TEST_model():
	"""
	Was supposed to test the individual components of the model.
	Defunct for now.
	"""
	return

	cap_input = np.array([[8, 214, 23, 1, 626, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
	model = FullModel(cap_input, "/Users/tejaswin.p/Downloads/epoch_13.hdf5")

	# import ipdb
	# ipdb.set_trace()

	x = model.preprocess_image("/Users/tejaswin.p/Downloads/eating_pizza_in_park.jpg")
	print x.shape

	# x = np.moveaxis(np.moveaxis(x, 1, 0), 2, 1)
	# x = np.expand_dims(x, axis=0)
	# print x.shape

	print "CORRECT caption similarity:", model.predict(x)

	model.caption = np.array([[8, 199, 23, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
	print "RANDOM caption similarity:", model.predict(x)

	K.clear_session()

def TEST_lime():
	model = FullModel(
		rnn_model_path="/Users/tejaswin.p/Downloads/epoch_9.hdf5", 
		word_index_path="/Users/tejaswin.p/Downloads/DICT_word_index.VAL.pkl", 
		vgg_weights_path="/Users/tejaswin.p/projects/devise-keras/vgg16_weights_th_dim_ordering_th_kernels.h5"
	)

	mask = model.run_lime(
		image_url="http://www.aacounty.org/sebin/n/m/dogpark.jpg", 
		caption_string="dog in the park"
	)

	assert mask.shape == (224, 224), "--TEST FAILED. Image size incorrect.--"

	K.clear_session()

if __name__ == '__main__':
	TEST_lime()

