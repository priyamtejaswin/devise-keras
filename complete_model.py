import numpy as np 
from extract_features_and_dump import define_model
from rnn_model import build_model, hinge_rank_loss
import keras 


def get_full_model(vgg_wts_path, rnn_wts_path):

	vgg              = define_model(vgg_wts_path) # get image->4096 model
	caption_features = Input(shape=(MAX_SEQUENCE_LENGTH,), name="caption_feature_input") # the caption input 
	full_model       = build_model(vgg.output, caption_features) # caption+4096Feats -> loss function 

	full_model.load_weights(rnn_wts_path, by_name=True)

	return full_model 

