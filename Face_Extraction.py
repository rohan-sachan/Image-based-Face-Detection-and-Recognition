import cv2
from mtcnn import MTCNN


def extract_face():
	detector = MTCNN()
	src = cv2.imread('captured.png')
	image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
	result = detector.detect_faces(image)
	if(len(result)==1):            
		x1, y1, width, height = result[0]['box']
		x2, y2 = x1 + width, y1 + height
		face_boundary = image[y1:y2, x1:x2]
		face_image = cv2.resize(face_boundary, (224,224))    
		cv2.imwrite('captured.png', cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
		print('Detected 1 face')
		return True, False
	else:
		print(f'Detecting {len(result)} faces, try again!!!')
		return False, True



