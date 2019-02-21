import pytesseract
import cv2


class Preprocessing:
	def __init__(self, input_img):
		self.img = input_img
		self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

		if cv2.mean(self.img)[0] < 128:
						self.img = cv2.bitwise_not(self.img)


	def natural_img_processing(self, noise):
		if noise:
				blur = cv2.medianBlur(self.img, 5) #noise filtering
		else:
				blur = self.img 
		r_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2) #Binarisation
		return r_img ## result img


	def digital_img_processing(self):
		#r_img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
		return self.img	
		pass
