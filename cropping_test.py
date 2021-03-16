#!/bin/python

from new_cropping import *
from PIL import Image
from tqdm import tqdm 
import os

def make_test():
	#example coordinates of a square
#top left, bottom left, bottom right, top right
	x = [20, 20, 40, 40]
	y = [20, 40, 40, 20]

#the following is stack overflow code to make a matrix of a bunch of zeroes and
#a square of ones which we designated by the two lists above
	contours = np.stack((x, y), axis = 1)
	polygon = np.array([contours], dtype = np.int32)
	zero_mask = np.zeros((1000, 1000), np.uint8)
	polyMask = cv.fillPoly(zero_mask, polygon, 255)
	
	return polyMask
	
matrix = make_test()
cv.imwrite("cropping_test.png", polyMask)

n = 150
cropper("./cropping_test.png", matrix, n, extension = ".png")

def assert_dimensions(output_img, n):
	
	img_path = os.path.abspath(output_img)
	image = Image.open(img_path)
	shape = image.size
	image.close()
	size_right = None
	if shape[0] == n and shape[1] == n:
		size_right = True
		
		print("Success: Output file", output_img, "has the correct dimensions")
		
	elif size_right == None:
		print("Error: Output file", output_img, "has incorrect dimensions")
		
	

	
assert_dimensions("cropping_test_cropped.png", n)