import cv2
import tensorflow as tf

CATEGORIES=["rashes","pimples"]

def prepare(filepath):
	IMG_SIZE =35
	img_array = cv2.imread(filepath)
	new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
	return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)


model = tf.keras.models.load_model('accurate_skin.model')

prediction =model.predict([prepare('acne93.jpg')])

print(prediction)


