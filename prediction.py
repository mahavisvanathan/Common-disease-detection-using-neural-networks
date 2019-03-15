import cv2
import tensorflow as tf

CATEGORIES=["jaundice","pinkeye","normal eye"]

def prepare(filepath):
	IMG_SIZE =35
	img_array = cv2.imread(filepath)
	new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
	return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)


model = tf.keras.models.load_model('accurate_eye.model')

prediction =model.predict([prepare('redeye.jpg')])
if prediction[0][0]==1:
	print("jaundice")
elif prediction[0][1]==1:
	print('pink eye')
else:
	print('normal eye')