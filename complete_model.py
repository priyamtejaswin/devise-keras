import numpy as np 
from extract_features_and_dump import define_model
from rnn_model import hinge_rank_loss
import keras 
from keras.models import Model, load_model 
from keras.layers import Input


def build_model(image_features, caption_features, embedding_matrix):
	
	# visual model 	
	# 	Hidden Layer - 1	
	image_dense1 = Dense(WORD_DIM, name="image_dense1")(image_features)
	image_dense1 = BatchNormalization()(image_dense1)
	image_dense1 = Activation("relu")(image_dense1)
	image_dense1 = Dropout(0.5)(image_dense1)	
	
	#   Hidden Layer - 2
	image_dense2 = Dense(WORD_DIM, name="image_dense2")(image_dense1)
	image_output = BatchNormalization()(image_dense2)

	# rnn model
	# embedding_matrix = pickle.load(open("KERAS_embedding_layer.TRAIN.pkl"))

	cap_embed = Embedding(
		input_dim=embedding_matrix.shape[0],
		output_dim=WORD_DIM,
		weights=[embedding_matrix],
		input_length=MAX_SEQUENCE_LENGTH,
		trainable=False,
		name="caption_embedding"
		)(caption_features)

	lstm_out_1 = LSTM(300, return_sequences=True)(cap_embed)
	lstm_out_2 = LSTM(300)(lstm_out_1)
	caption_output = Dense(WORD_DIM, name="lstm_dense")(lstm_out_2)
	caption_output = BatchNormalization()(caption_output)

	concated = concatenate([image_output, caption_output], axis=-1)

	return concated

def get_full_model(vgg_wts_path, rnn_wts_path):

	vgg              = define_model(vgg_wts_path) # get image->4096 model
	
	caption_features = Input(shape=(MAX_SEQUENCE_LENGTH,), name="caption_feature_input") # the caption input 
	embedding_matrix = pickle.load(open("KERAS_embedding_layer.TRAIN.pkl"))
	
	full_model       = build_model(vgg.output, caption_features, embedding_matrix=embedding_matrix) # caption+4096Feats -> loss function 
	
	completeModel    = Model(inputs=[vgg.input, caption_features], outputs=full_model.output)

	return full_model 

