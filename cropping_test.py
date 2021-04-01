#!/bin/python
import collections, functools, operator
from new_cropping import *
from PIL import Image
from tqdm import tqdm 
import os
import cv2 as cv
import time
import sys

def make_test(m, n, a, b, color=1):
	
	x = [a, a, b, b]
	y = [a, b, b, a]

	contours = np.stack((x, y), axis = 1)
	polygon = np.array([contours], dtype = np.int32)
	zero_mask = np.zeros((m, n), np.uint8)
	polyMask = cv.fillPoly(zero_mask, polygon, color)
	
	return polyMask
	

def assert_dimensions(output_img, n):
	
	img_path = os.path.abspath(output_img)
	image = Image.open(img_path)
	shape = image.size
	image.close()
	size_right = False
	if shape[0] == n and shape[1] == n:
		size_right = True
		
		print("Success: Output file", output_img, "has the correct dimensions")
		
	elif size_right == False:
		print("Error: Output file", output_img, "has incorrect dimensions")


masks = "/Users/brie/Desktop/ancientImages/petrousPredictions/"
inputs = "/Users/brie/Desktop/ancientImages/petrousImages/"

fname = "/Users/brie/Desktop/petrous_files.txt"
with open(fname, "r") as fd:
    images = fd.read().splitlines()

# images = images[1501:1900]
   
for f in images:
	
	print(f)
	file = masks + f[:-4] + "_prediction.png"
	image = cv.imread(file)
	img = np.array(image)
	
	matrix = img[:,:,1]
	
	z = cropper(inputs, f, matrix, out_dir = "/Users/brie/Desktop/ancientImages/croppedOutputs/")