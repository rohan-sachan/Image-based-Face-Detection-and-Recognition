import tensorflow as tf
import csv
import pandas as pd
import numpy as np

def write_to_csv(name, embedding):
	embedding = embedding.flatten()
	emb2 = embedding.tolist()
	emb1 = [name]
	emb = emb1 + emb2
	with open('database.csv', "a", newline="") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(emb)
		
def read_names_from_csv():
	data = pd.read_csv("database.csv", header = None)
	names = data.iloc[:,0].astype(str)
	names = names.values.tolist()
	return names

def read_emb_from_csv():
	data = pd.read_csv("database.csv", header = None)
	emb_stored = data.iloc[:,1:]
	emb_nparray = emb_stored.to_numpy()
	emb_nparray.flatten()
	return emb_nparray

def prediction(img_emb, predictor):
	# from keras_facenet import FaceNet
	# predictor = tf.keras.models.load_model('modelfnbest.h5')
	# embedder = FaceNet()

	img_emb = img_emb.flatten()
	emb_stored = read_emb_from_csv()
	names = read_names_from_csv()
	index_number = 0
	position = 0
	prediction_prob = []
	for row in emb_stored:
		x = np.subtract(row, img_emb)
		x = np.absolute(x)
		x = np.expand_dims(x, axis=0)
		val = predictor.predict(x)
		prob = val[0]
		prediction_prob.append(prob)
	position = prediction_prob.index(max(prediction_prob))
	if (prediction_prob[position]>0.90) :
		return names[position]
	else:
		return "Unauthenticated Person"

