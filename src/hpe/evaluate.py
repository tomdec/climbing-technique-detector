from cv2 import cvtColor, COLOR_BGR2RGB

def predict_landmarks(image, model):
	image_height, image_width, _ = image.shape

	image = cvtColor(image, COLOR_BGR2RGB) 		# COLOR CONVERSION BGR 2 RGB
	image.flags.writeable = False				# Image is no longer writeable
	
	results = model.process(image)				# Make prediction
	
	image.flags.writeable = True				# Image is now writeable 
	image = cvtColor(image, COLOR_BGR2RGB)		# COLOR COVERSION RGB 2 BGR

	return image, results, (image_height, image_width)