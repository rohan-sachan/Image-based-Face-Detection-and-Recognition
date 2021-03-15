import tensorflow as tf
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
import cv2
from Face_Extraction import *
from database_utils import *
import numpy as np
from keras_facenet import FaceNet

def start_recog():
	recog = Toplevel()
	recog.title('Face Recognizer')
	back_button = Button(recog, text = 'Back', padx = 20, pady = 20, command = lambda: (recog.destroy(), cam.release()))
	back_button.grid(row = 0, column = 0, sticky = W)
	cam = cv2.VideoCapture(0)
	cv2.namedWindow("Capture")
	flag = True
	response = 0
	while not response:
		while flag:
			ret, frame = cam.read()
			if not ret:
				print("failed to grab frame")
				break	
			cv2.imshow("Capture", frame)

			k = cv2.waitKey(1)
			if k%256 == 32:
				# SPACE pressed
				img_name = "captured.png"
				cv2.imwrite(img_name, frame)
				print("{} written!".format(img_name))
				flag = False
				
		cv2.destroyAllWindows()	
		img = ImageTk.PhotoImage(Image.open('captured.png'))
		img_shower = Label(recog, image = img)
		img_shower.image = img
		img_shower.grid(row = 1, column = 0)
		response = messagebox.askyesno('Alert!!', 'Is the image ok?')
		if not response:
			flag = True
			img_shower.grid_forget()
		else:	
			response, flag = extract_face()

	cam.release()
	cv2.destroyAllWindows()
	img = Image.open('captured.png')
	img = np.array(img)
	img = np.reshape(img, [1, 224, 224, 3])
	embeddings = embedder.embeddings(img)
	ans = prediction(embeddings, predictor)
	name_label = Label(recog, text = ans, padx = 20, pady = 30)
	name_label.grid(row = 3, column = 0)
	exit_button = Button(recog, text = 'Exit', padx = 20, pady = 20, command = recog.destroy)
	exit_button.grid(row = 4, column = 0)

def add_face():
	new_face = Toplevel()
	new_face.title('Face Recognizer')
	back_button = Button(new_face, text = 'Back', padx = 20, pady = 20, command = lambda: (new_face.destroy(), cam.release()))
	back_button.grid(row = 0, column = 0, sticky = W)
	cam = cv2.VideoCapture(0)
	cv2.namedWindow("Capture")
	flag = True
	response = 0
	while not response:
		while flag:
			ret, frame = cam.read()
			if not ret:
				print("failed to grab frame")
				break	
			cv2.imshow("Capture", frame)

			k = cv2.waitKey(1)
			if k%256 == 32:
				# SPACE pressed
				img_name = "captured.png"
				cv2.imwrite(img_name, frame)
				print("{} written!".format(img_name))
				flag = False
				
		cv2.destroyAllWindows()	
		img = ImageTk.PhotoImage(Image.open('captured.png'))
		img_shower = Label(new_face, image = img)
		img_shower.image = img
		img_shower.grid(row = 1, column = 0)
		response = messagebox.askyesno('Alert!!', 'Is the image ok?')
		if not response:
			flag = True
			img_shower.grid_forget()
		else:	
			response, flag = extract_face()

	

	cam.release()
	cv2.destroyAllWindows()
	ques = Label(new_face, text = 'Enter the name of the user', padx = 50)
	name_box = Entry(new_face, width = 50)
	
	def click():
		name = name_box.get()
		img = Image.open('captured.png')
		img = np.array(img)
		img = np.expand_dims(img, axis = 0)
		img = np.vstack([img])
		embeddings = embedder.embeddings(img)
		write_to_csv(name, embeddings)
		new_face.destroy()

	enter_button = Button(new_face, text = 'Enter name', padx = 50, command = click)		
	ques.grid(row = 2, column = 0)
	name_box.grid(row = 3, column = 0)
	enter_button.grid(row = 4, column = 0)
	
def database():
	data = pd.read_csv("database.csv", header = None)
	names = data.iloc[:,0].astype(str)
	names = names.values.tolist()
	string = ''
	for name in names:
		string+=name + '\n'

	data = Toplevel()
	data.title('Database')
	names = Label(data, text = string, padx = 20, pady  =20)
	names.grid(row = 0, column = 0)
	exit_button = Button(data, text = 'Exit', padx = 20, pady = 20, command = data.destroy)
	exit_button.grid(row = 1, column = 0)	


root = Tk() 
embedder = FaceNet()
predictor = tf.keras.models.load_model('modelfnbest.h5')
root.title('Face Recognizer')
recog_button = Button(root, text = '1) Start Recognition', padx = 55, pady = 30, command = start_recog)
data_button = Button(root, text = '2) Add new face', padx = 65, pady = 30, command = add_face)
database_button = Button(root, text = '3) Database', padx = 80, pady = 30, command = database) 
recog_button.grid(row = 0, column = 0, sticky = W+E)
data_button.grid(row = 1, column = 0, sticky = W+E)
database_button.grid(row = 2, column = 0, sticky = W+E)
button_quit = Button(root, text = '4) Exit', padx = 90, pady = 30, command = root.destroy)
button_quit.grid(row = 3, column = 0, sticky = W+E)
root.mainloop()