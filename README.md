# Interpretable Image Search
*- by Priyam Tejaswin and Akshay Chawla*
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
│   ├── 12345.jpg
│   ├── 32561.jpg
│   ├── 45321.jpg
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
	1. vgg16 pre-trained weights: 
	2. word index mappings: 
	3. Pre-trained DeVISE weights: 
	4. MS COCO captions: 
	5. cache.h5: 

2. Clone this repository to your local system. 
3. Run the server using: 

```
python server_lime_contours.py --word_index=/path/to/devise_cache/DICT_word_index.VAL.pkl --cache=/path/to/devise_cache/cache.h5 --model=/path/to/devise_cache/epoch_13.hdf5 --threaded=0 --host=127.0.0.1 --port=5000 --dummy=0 --captions_train=/path/to/devise_cache/annotations/captions_train2014.json --captions_valid=/path/to/devise_cache/annotations/captions_val2014.json --vgg16=/path/to/devise_cache/vgg16_weights_th_dim_ordering_th_kernels.h5
```


4. Be careful to replace /path/to/devise_cache/ to the correct path to your devise_cache folder.
5.  Open a modern web browser (we tested this on firefox quantum 57) and navigate to localhost:5000 to view the webpage.
