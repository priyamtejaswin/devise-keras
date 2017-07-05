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
	unzip glove.6B.zip
fi
echo

uiucSource="SOURCE_PASCAL_SENTENCES_vision.cs.uiuc.edu.html"
if [ -f "$uiucSource" ];
        then
        echo -e "$uiucSource exists.\n"
else
        echo -e "$uiucSource not found. Downloading...\n"
        wget -O $uiucSource "vision.cs.uiuc.edu/pascal-sentences/"
fi
echo

featsDir="processed_features"
if [ -d "$featsDir" ];
	then
	echo -e "$featsDir exists.\n"
else
	mkdir $featsDir
	echo -e "Created $featsDir\n"
fi

snapsDir="snapshots"
if [ -d "$snapsDir" ];
	then
	echo -e "$snapsDir exists.\n"
else
	mkdir $snapsDir
	echo -e "Created $snapsDir\n"
fi

logsDir="logs"
if [ -d "$logsDir" ];
        then
        echo -e "$logsDir exists.\n"
else
        mkdir $logsDir
        echo -e "Created $logsDir\n"
fi

echo -e "DONE\nFollow steps listed under SETUP in README.md"
