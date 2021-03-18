#importing libraries
import collections, functools, operator
import cv2 as cv
from PIL import Image
import numpy as np
import json
import os

#given a filename and vgg formattted json file path from makesense.ai or other image annotation software,
#this function iterates through the json file and returns the x and y coordinates as lists
def coordinates_from_json(filename, json_path):
    file = open(json_path)
    data = json.load(file)
    file.close() #we now have the json object as a python dictionary
    x_list = data[filename]['regions']['0']['shape_attributes']['all_points_x']
    y_list = data[filename]['regions']['0']['shape_attributes']['all_points_y']
    return x_list, y_list


#the following function takes a filename of an image, a json object containing coordinate information,
#and a directory (str) in which to store the resulting output, the mask
def mask_from_file(image_dir, filename, json_path, mask_dir, n=1):
    x_list, y_list = coordinates_from_json(filename, json_path)
    path = os.path.abspath(image_dir + '/' + filename)
    image = Image.open(path)
    shape = image.size
    #print(shape)
    image.close()
    contours = np.stack((x_list, y_list), axis = 1)
    polygon = np.array([contours], dtype = np.int32)
    zero_mask = np.zeros((shape[1], shape[0]), np.uint8)
    polyMask = cv.fillPoly(zero_mask, polygon, n)
    cv.imwrite(mask_dir + '/' + filename[:-4] + '_mask.png', polyMask)
    return polyMask

# This finds the x and y coordinates of the start and end positions, used to find the centroid of the image
# Returns a dictionary containing the start position, and the end position
# To be used with simpleCentroid function

def get_coordinates(matrix):
	
	coordinates = dict()
	first_one = False
	
	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
		
			if matrix[i][j] == 1 and first_one == False:
				coordinates['start'] = (i, j)
				first_one = True
			
			if matrix[i][j] == 1 and first_one == True:
				coordinates['end'] = (i, j) # i = row, j = column
			
	start_end_coordinate_dict = coordinates
	
	return start_end_coordinate_dict

# Find the x and y coordinates of the center of the polygon
# Returns a dictionary that contains these coordinates
def simpleCentroid(matrix, start_end_coordinate_dict = None):
	
	def coordinates2centroid(start_end_coordinate_dict):
		
		coords = dict()
		y = None
		x = None
		
		coordinates = start_end_coordinate_dict
		
		if coordinates['end'][0] > coordinates['start'][0]:
			y = coordinates['end'][0] - coordinates['start'][0]
			coords['y'] = coordinates['start'][0]
		elif coordinates['end'][0] < coordinates['start'][0]:
			y = coordinates['start'][0] - coordinates['end'][0]
			coords['y'] = coordinates['end'][0]
			
		if coordinates['end'][1] > coordinates['start'][1]:
			x = coordinates['end'][1] - coordinates['start'][1]
			coords['x'] = coordinates['start'][1]
		elif coordinates['end'][1] < coordinates['start'][1]:
			x = coordinates['start'][1] - coordinates['end'][1]
			coords['x'] = coordinates['end'][1]
			
		midy = y // 2
		midx = x // 2
		
		y = coords['y'] + midy
		x = coords['x'] + midx
		
		return ({'x': x, 'y': y})
	
	coords = None
	start_end_positions = None	
	if start_end_coordinate_dict != None:
		coords = coordinates2centroid(start_end_coordinate_dict)
	elif start_end_coordinate_dict == None:
		start_end_positions = get_coordinates(matrix)
		coords = coordinates2centroid(start_end_positions)
			
	return coords
		

def n_finder(matrix, n = 1000, j = 200, k = 400):

	indexer_dict_list = []
	indexer_lengths = []
	
	for i in range(len(matrix)):
		
		indexer_dict = dict()
		indexer_list = []
		
		for j in range(len(matrix[i])):
			
			if matrix[i][j] == 1:
				indexer_dict[j] = 1
				indexer_list.append(1)
				
		indexer_dict_list.append(indexer_dict)
		indexer_lengths.append(len(indexer_list))
		
	
	result = dict(functools.reduce(operator.add, map(collections.Counter, indexer_dict_list)))
	vertical_index = max(result.items(), key=operator.itemgetter(1))[0]
	
	vertical_value = result[vertical_index]
		
	horizontal_value = max(indexer_lengths)
		
	maximum_value = max([vertical_value, horizontal_value])
		
	new_n = None
	if maximum_value >= n:
		new_n = maximum_value + j
	elif maximum_value < n:
		new_n = maximum_value + k
				
	return new_n

	
#given the filepath of an image (in .jpg) and its corresponding mask of ones and zeros,
#this function crops and stores the image in the same directory
#where n is half the length of the desired width and height
def cropper(image_dir, filename, matrix, n = None, extension = ".jpg"):

    path = os.path.abspath(image_dir + '/' + filename)
    img = cv.imread(path)
    
    n = n_finder(matrix) if n == None else n_finder(matrix, n)
        
    coords = simpleCentroid(matrix)
	
    center_x, center_y = coords['x'], coords['y']

    left_bound = center_x - n // 2 if (center_x - n // 2) >= 0 else 0
    right_bound = center_x + n // 2 if (center_x + n) < len(matrix[0]) else (len(matrix[0]) - 1)
    bottom_bound = center_y + n // 2 if (center_y + n) < len(matrix) else len(matrix) - 1
    top_bound = center_y - n // 2 if (center_y - n) >= 0 else 0

    cropped_img = img[top_bound:bottom_bound, left_bound:right_bound]
    cropped_path = path[0:(len(path) - 4)] + "_cropped" + extension

    cv.imwrite(cropped_path, cropped_img)
	
#iterates through files in a folder to create masks (takes files from filepath and json annotations from annotations and places masks in mask_dir)
def main(filepath, annotations, mask_dir):
    #cnt = 0
    for f in os.listdir(filepath):
        # print(f)
        #cnt += 1
        #xList, yList = coordinates_from_json(f, annotations)
        #print(cnt)
        mask_from_file(filepath, f, annotations,mask_dir)

#PA - Petrous Annotations
#TA - Teeth Annotations
#A - Alaukik
#K - Kushal
file_pathAPA = r"C:\Users\Owner\OneDrive - The University of Texas at Austin\Narasimhan Lab\Bone Image Annotation\AlaukiksPA"
file_pathATA = r"C:\Users\Owner\OneDrive - The University of Texas at Austin\Narasimhan Lab\Bone Image Annotation\AlaukiksTA"
file_pathKPA = r"C:\Users\Owner\OneDrive - The University of Texas at Austin\Narasimhan Lab\Bone Image Annotation\KushalsPA"
file_pathKTA = r"C:\Users\Owner\OneDrive - The University of Texas at Austin\Narasimhan Lab\Bone Image Annotation\KushalsTA"

mask_dirAPA = r"C:\Users\Owner\OneDrive - The University of Texas at Austin\Narasimhan Lab\Bone Image Annotation\APA_Mask"
mask_dirKPA = r"C:\Users\Owner\OneDrive - The University of Texas at Austin\Narasimhan Lab\Bone Image Annotation\KPA_Mask"
mask_dirATA = r"C:\Users\Owner\OneDrive - The University of Texas at Austin\Narasimhan Lab\Bone Image Annotation\ATA_Mask"
mask_dirKTA = r"C:\Users\Owner\OneDrive - The University of Texas at Austin\Narasimhan Lab\Bone Image Annotation\KTA_Mask"

main(file_pathAPA, 'PA.json', mask_dirAPA)
main(file_pathKPA, 'PK.json', mask_dirKPA)
main(file_pathATA, 'TA.json', mask_dirATA)
main(file_pathKTA, 'TK.json', mask_dirKTA)
