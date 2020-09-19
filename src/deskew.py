import numpy as np
import cv2


class Deskew:
		def __init__(self, input_img):
				self.input_img = input_img


		def calculator_angle(self):
				# --gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
				gray = cv2.bitwise_not(self.input_img)

				# !@#$%^&*()
				thresh = cv2.threshold(gray, 0, 255, 
						cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

				coords = np.column_stack(np.where(thresh > 0))
				angle = cv2.minAreaRect(coords)[-1]
				
				if angle < -45:
						angle = -(90 + angle)
				else:
						angle = -angle
				
				return angle


		def rotation_img(self, angle):
				# rotate the image to deskew it
				(h, w) = self.input_img.shape[:2]
				center = (w // 2, h // 2)
				M = cv2.getRotationMatrix2D(center, angle, 1.0)
				rotated = cv2.warpAffine(self.input_img, M, (w, h),
					flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
				
				return rotated

		def run(self):
				#if self.input_img:
				return self.rotation_img(self.calculator_angle())
