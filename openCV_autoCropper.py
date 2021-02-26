#!/usr/bin/python

import cv2
from PIL import Image
import os 

fdict = {"RSIP_Example_HipSegmentation.jpg" : 1}

def autoCropper(image_dir, cropped_dir, file_dict):
	full_path = os.path.abspath(image_dir)
	full_cropped_path = os.path.abspath(cropped_dir)
	
		
	for k, v in file_dict.items():
		
		fname = full_path + "/" + k
		
		cropped_fname = full_cropped_path + "/" + k[:-4] + "_cropped.jpg"
				
		absolute_img_filename = fname		
		image = Image.open(absolute_img_filename)
		shape = image.size
		image.close()
			
		x = shape[0]
		y = shape[1]
		midx = x // 2
		midy = y // 2

		img = cv2.imread(fname)
		
		if v == 1:
			crop_img = img[0:midy, midx:x]
			
			
		elif v == 2: 
			crop_img = img[0:midy, 0:midx]
		
		elif v == 3: 
			crop_img = img[midy:y, 0:midx]
		
		elif v == 4: 
			crop_img = img[midy:y, midx:x]
			
		
		cv2.imwrite(cropped_fname, crop_img)
		
		return print(cropped_fname, "created successfully")
			

			
crop = autoCropper("test_images_dir", "crop_test_dir", fdict)
		

	
