# Interpretable Image Search
*- by Priyam Tejaswin and Akshay Chawla*

**LIVE WEB DEMO AT:** http://35.193.106.36:5050/  . This will forward to a GCP instance endpoint. If you're having issues accessing it from your internal work or office network, please raise an issue or contact [Priyam](mailto:priyamtejaswin@gmail.com), [Akshay](mailto:chawla.akshay1234@gmail.com).

## Introduction
This project extends the original [Google DeViSE](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41473.pdf) paper to create a functioning image search engine with a focus on interpreting search results. We have extended the original paper in the following ways. First, we added an RNN to process variable length queries as opposed to single words. Next, to understand how the network responds to different parts of the query(like noun phrases) and the image, we leverage [Ribeiro et.al's LIME](https://arxiv.org/pdf/1602.04938v1.pdf) for model-agnostic interpretability. It has been tested on subsets of the [UIUC-PASCAL dataset](http://vision.cs.uiuc.edu/pascal-sentences/) and the final network has been trained on the [MSCOCO 2014 dataset](http://cocodataset.org/#home).

## Codebase
**LIME** is the main deployment branch for this project. The code is organised as follows.
```
devise-keras/
├── build_cache.py
├── cache_lime.py
├── complete_model.py
├── contour_utils.py
├── extract_features_and_dump.py
├── gpu0.sh
├── gpu1.sh
├── gpu2.sh
├── lime_results_dbase.db
├── nlp_stuff.py
├── README.md
├── requirements.txt
├── rnn_model.py
├── server_lime_contours.py
├── server_nolime.py
├── server.py
├── simplified_complete_model.py
├── static
│   ├── <SAMPLE_IMAGES>.jpg
│   ├── jquery-1.12.4.js
│   ├── jquery-ui.js
│   ├── lime_queries.json
│   ├── myscript.js
│   ├── myscript_lime_contours.js
│   ├── myscript_nolime.js
│   └── overlays_cache
│       ├── <CACHED_LIME_IMAGES>.png
├── templates
│   ├── index.html
│   ├── index_lime_contours.html
│   └── index_nolime.html
├── tensorboard_logging.py
└── validation_script.py
```

## Development History
**master**: This repository houses code to replicate the base DeViSE paper. Since the project - and it’s scope - has grown organically as we advanced, we decided to branch out and leave the vanilla DeViSE code base intact. The **master** branch contains code to setup the experiment, download and pre-process data for implementing the paper. `model.py` contains the code for the model. Due to computational constraints, the experiments are run on the UIUC-PASCAL sentences dataset as opposed to ImageNet. This dataset contains 50 images per category for 20 categories along with 5 captions per image. These captions are used for the extension of the project in the **devise-rnn** branch.

**devise-rnn**: This branch is an extension of the **master** codebase. **rnn_model.py** contains the code for the extended model. Due to computational constraints, the experiments are run on the UIUC PASCAL sentences dataset. This dataset contains 50 images per category along with 5 captions per image. These captions are used in the same pairwise loss function to learn an image search model entirely from annotated images.

**mscoco-search**: This branch extends **devise-rnn**. It contains updates and changes for training the model on the MSCOCO 2014 dataset.

**ui**: This branch contains code for building the user-interface of the search engine and the backend for running the model.

**LIME**: This branch extends **mscoco-search** and includes a frontend from **ui**. Additionally, it adds interpretability modules.

## How to run
1. First download these required files and place them in a folder called devise_cache. 
	1. vgg16 pre-trained weights: https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	2. word index mappings: https://www.dropbox.com/s/h9m7ju42sckehy5/DICT_word_index.VAL.pkl?dl=0
	3. Pre-trained DeVISE weights: https://www.dropbox.com/s/7lsubnf9fna7kun/epoch_13.hdf5?dl=0
	4. MS COCO captions: http://cocodataset.org/#download
	5. cache.h5: https://www.dropbox.com/s/xoza70y5zyh5d99/cache.h5?dl=0

2. Clone this repository to your local system. 
3. Run the server using: 

```
python server_lime_contours.py \
--word_index=/path/to/devise_cache/DICT_word_index.VAL.pkl \
--cache=/path/to/devise_cache/cache.h5 \
--model=/path/to/devise_cache/epoch_9.hdf5 \
--threaded=0 \
--host=127.0.0.1 \
--port=5000 \
--dummy=0 \
--captions_train=/path/to/devise_cache/annotations/captions_train2014.json \
--captions_valid=/path/to/devise_cache/annotations/captions_val2014.json \
--vgg16=/path/to/devise_cache/vgg16_weights_th_dim_ordering_th_kernels.h5
```


4. Be careful to replace /path/to/devise_cache/ to the correct path to your devise_cache folder.
5.  Open a modern web browser (we tested this on firefox quantum 57) and navigate to localhost:5000 to view the webpage.

## How it works

### DeViSE: A Deep Visual-Semantic Embedding Model
 In this paper, the authors present a new deep visual-semantic embedding model trained to identify visual objects using both labeled image data as well as semantic information gleaned from unannotated text. They accomplish this by minimizing a combination of the cosine similarity and hinge rank loss between the embedding vectors learned by the language model and the vectors from the core visual model as shown below. 
 
 ![devise_core](https://user-images.githubusercontent.com/8658591/34649979-1b47f090-f3df-11e7-8833-e488dc33cad0.PNG)

In the interest of time, we did not train a skip-gram model ourselves but chose to use the [GloVe (Global Vectors for Word Representation)](https://nlp.stanford.edu/projects/glove/) model from the stanford NLP group as our initilization. 

### Extending DeViSE for captions 
In order to encode variable length captions in the language model, we used an RNN network consisting of an Embedding input layer and 2 LSTM cells with 300 hidden units. This gave us an output vector that can be used for computing the similarity metric mentioned before. 

![RNN picture](https://user-images.githubusercontent.com/8658591/34650207-00c08814-f3e3-11e7-95f9-e2cf081661bd.PNG)

This allows us to map images and their captions to a common semantic embedding space. We use this feature to search for images in the embedding space that are close to the query entered by a user. 
 
### Adding LIME explainability 
In order to explain the relevance of our results, we modified Ribeiro et al's LIME such that it highlights salient regions relevant to the user's query. This gives visual cues about the regions in an image which maximally contributed to its retrieval. 

![lime_pic](https://user-images.githubusercontent.com/8658591/34650324-7ab4edd4-f3e5-11e7-9ffa-95798fbf5638.PNG)

### Deployment 
We deployed our work as an image search engine by building html, css and js components. Concretely, we run a server in the background that communicates with a frontend ui that displays the search results and lime saliency regions. 

The user enters a search query which is communicated to the server. The server runs the query string through the trained RNN model to find its final state vector. We search for the top 10 images closest to that query in the embedding space and return the links to those images. 

Once the retrieved images have been displayed on the webpage, we request the server to extract appropriate noun and verb phrases using a dependancy parser. These phrases are displayed as button on the webpage. We also request the server to fetch salient regions for each phrase and each returned image. Selecting a phrase button will highlight its approprate region in all images. 

**NOTE: Calculating LIME results for each (query, images) tuple requires ~3 hours as each phrase has to be run against every image retrieved. Hence, in the interest of time (and the limitations of having 1/0 GPUs) we pre-cache the LIME results for some sample queries. These sample queries can be accessed via clicking on the drop-down menu which appears when the user clicks on the search box.** While LIME results are available only for a limited set of queries, the search and retrieval sans lime works for all queries, provided the input tokes are present in our dictionary. 

Apologies to web designers, we just cannot write good html/css.

## Cats vs Dogs
Searching for the internet's two favourite pets.
### cat sitting on a tv
![Imgur](https://i.imgur.com/bLsYxMr.gifv)
