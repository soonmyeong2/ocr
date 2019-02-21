from imutils.object_detection import non_max_suppression # during the experiment
import numpy as np
import heapq
import cv2


class TextDetection:
		def __init__(self, input_img, origin_img):
				self.img = input_img
				self.origin_img = origin_img
				self.text_img = []
				self.text_xy = []


		def detection(self):
				small = cv2.pyrDown(self.img)
				#small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
				grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

				_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

				kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
				connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
				# using RETR_EXTERNAL instead of RETR_CCOMP / EXTERNAL
				# Switch to RETR_CCOMP using NMS
				# Useful when text area are ambiguous
				contours, hierarchy = cv2.findContours(connected.copy(), 
						cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
				

				mask = np.zeros(bw.shape, dtype=np.uint8)
				temp_img = self.img.copy() ###
				for idx in range(len(contours)):
						x, y, w, h = cv2.boundingRect(contours[idx])
						mask[y:y+h, x:x+w] = 0
						cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
						r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
		
						if r > 0.44 and w > 8 and h > 8:
							#debug								
								cv2.rectangle(temp_img, ((x-1)*2, (y-1)*2), ((x+w)*2, (y+h)*2), (0, 255, 255), 2)
								
								#print(x, x+w, y, y+h)
								heapq.heappush(self.text_xy, [y, x, y+h, x+w])
				while len(self.text_xy):
						y, x, yh, xw = heapq.heappop(self.text_xy)
						print(x, xw, y, yh)
						crop_img = self.img[y*2:yh*2, x*2:xw*2]
						
						if cv2.mean(crop_img)[0] < 128:
								crop_img = cv2.bitwise_not(crop_img)	
						self.text_img.append([crop_img, y])
	
				cv2.imshow('det', temp_img) # gray scale binary img
				cv2.waitKey()

				return self.text_img
