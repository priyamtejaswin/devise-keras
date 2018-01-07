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
