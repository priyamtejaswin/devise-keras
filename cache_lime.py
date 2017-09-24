import numpy as np
import matplotlib.pyplot as plt
import cv2 
import requests
import time 
from simplified_complete_model import FullModel 
import sqlite3 
import argparse

parser = argparse.ArgumentParser(description='utility to cache LIME results')
parser.add_argument("--query", type=str, help="query for LIME results", required=True)
parser.add_argument("--word_index", type=str, help="location of the DICT_word_index.VAL/TRAIN.pkl", required=True)
parser.add_argument("--model", type=str, help="location of the model.hdf5 snapshot", required=True)
parser.add_argument("--vgg16", type=str, help="location of vgg16 weights", required=True)
args = parser.parse_args()

# LIME Model
model = FullModel(
			rnn_model_path = args.model, 
			word_index_path = args.word_index, 
			vgg_weights_path = args.vgg16
		)

def dump_to_db(phrase, flickr_url, image_name):
	''' This dumps the phrase, flickr_url and image_name to lime_results_dbase.db database '''
	conn = sqlite3.connect("lime_results_dbase.db")
	cursor = conn.cursor()
	_exec_str = "insert into results values( '{}' , '{}' , '{}' )".format(phrase, flickr_url, image_name)
	print "Running: ", _exec_str
	cursor.execute(_exec_str)
	conn.commit()
	conn.close()


def main():

	user_query = args.query
	print "User Searched for: ", user_query

	# flickr images
	resp = requests.get(url="http://127.0.0.1:5000/_process_query", params={"query":user_query}) # get response from server 
	assert resp.ok == True, "bad response from server, please check if server is running"
	coco_urls   = resp.json()["coco_urls"]
	flickr_urls = resp.json()["flickr_urls"]
	flickr_urls = [str(k) for k in flickr_urls]
	captions    = resp.json()["captions"]

	print "Flickr Urls: "
	print flickr_urls

	# query -> phrases
	resp = requests.get(url="http://127.0.0.1:5000/_get_phrases", params={"query":user_query}) # get response from server 
	assert resp.ok == True, "bad response from server, please check if server is running"
	phrases = resp.json()["phrases"]
	phrases = [str(k) for k in phrases]
	phrases = [phrase.replace("_"," ") for phrase in phrases]

	# import ipdb; ipdb.set_trace()

	# Run LIME for (phrase, flickr_url) tuple
	for url in flickr_urls:
		for phrase in phrases:
			
			print "Image:", url, " | phrase: ", phrase 

			mask = model.run_lime(
						image_url=url, 
						caption_string=phrase
					)

			mycolor = np.array([240, 10, 10])
			explain_im   = np.ones((224,224,3)).astype(np.uint8) * 255
			explain_im[mask==1.0] = mycolor

			# save explain_im to disk
			imname = str(time.time()) + ".png"
			cv2.imwrite("static/overlays_cache/"+imname, explain_im)

			dump_to_db(phrase, url, imname)


if __name__ == '__main__':
	main()
