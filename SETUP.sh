#!/usr/bin/env bash
## Script to download and setup data.

echo -e "\nRUNNING SETUP.sh\n\n"

vggFile="vgg16_weights_th_dim_ordering_th_kernels.h5"
if [ -f "$vggFile" ];
	then
	echo -e "$vggFile exists.\n"
else
	echo -e "$vggFile not found. Downloading...\n"
	wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
fi
echo

gloveFile="glove.6B.50d.txt"
if [ -f "$gloveFile" ];
	then
	echo -e "$gloveFile exists.\n"
else
	echo -e "$gloveFile not found. Downloading...\n"
	wget http://nlp.stanford.edu/data/glove.6B.zip
	echo -e "Extracting...\n"
	tar -xvzf glove.6B.zip
fi
echo

echo -e "DONE\n"