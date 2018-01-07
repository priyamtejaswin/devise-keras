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
