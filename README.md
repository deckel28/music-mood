# music-mood
Music mood recognition using various Audio features from a song
## Model Demo on Google Colab
#### mood_train.ipynb
	Training Notebook
#### mood_pred.ipynb
	Evaluation and Testing Notebook
#### word2vec.ipynb
	An attempt to cluster the wide-ranging tags using Google News word embeddings
## configs
	Set the default parameters for feature extraction 
## src 
#### tags.py
	One-Hot encodes labels for songs from metadata
#### getfeatures.py
	Contains the classes for extracting features from all the songs in a song directory.
#### train.py
	Extract Features
	Labels Encoding
	Defining Model and Training
#### eval.py
	Predictions 
	Performance Visualisation
#### test.py
	Experiment with songs and print predicted tags
## utils
	Miscellaneous scripts to pre-preprocess the data
