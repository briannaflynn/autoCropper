#importing libraries
import collections, functools, operator
import cv2 as cv
from PIL import Image
import numpy as np
import json
import os
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        now = time.time()
        retval = func(*args, **kwargs)
        print('{} took {:.5f}s'.format(func.__name__, time.time() - now))
        return retval
    return wrapper

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


@timeit
def numpy_centroid(matrix):
	
	rows = [i for i in range(len(matrix))] # get a list of indexes for all rows of the matrix
	
	columns = [j for j in range(len(matrix[0]))] # get a list of indexes for all columns of the matrix
	
	# convert the lists to numpy arrays
	r = np.array(rows)
	c = np.array(columns)
	
	def value_max_width_len(values):
	
		j = values[np.fromiter(map(len, values), int).argmax()] # use this to get the longest array from an array of arrays
		#v = max(map(len, values))
		return j

	def numpy2centroid(matrix, index_array):
		
		mat = index_array * matrix # broadcast the array of indexes to all the arrays of the matrix, multiply together
		# will get matrix of 0s and the indexes
		
		
		mat = [n[n != 0] for n in mat] # remove all zeros, so now we have a matrix of arrays with either only indexes, or empty arrays
		# convert the list back into an array
		matrix = np.array(mat)
		
		# x is now the longest array of the matrix (remember, without zeros, the lengths of the index arrays will differ based on the length of the polygon)
		x = value_max_width_len(matrix)
		j = x[-1] - x[0] # subtract the last index from the first index
		
		center = x[0] + j // 2 # divide the difference between the indexes in half, and add that to first index to find the center
		
		return center
		
	x_coord = numpy2centroid(matrix, c) # this will give you the x coordinate of the centroid, from our calculations using the row indexes
	trans_matrix = np.transpose(matrix) # transpose the matrix, so that the columns become the rows, and the rows become the columns
	y_coord = numpy2centroid(trans_matrix, r) # use the transposed matrix and the column index array to find the y coordinates of the centroid
	
	return {'x': x_coord, 'y': y_coord} # returns a dictionary with the x and y coordinates
		


#TODO: rewrite n_finder using the code for numpy_centroid
# basically the code as numpy_centroid, instead of get center get the longest values, get the max of x and y longest values, then do lines 189 through 193

@timeit
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
@timeit
def cropper(image_dir, filename, matrix, n = 1000, extension = ".jpg"):

    path = os.path.abspath(image_dir + '/' + filename)
    img = cv.imread(path)
    
    #n = n_finder(matrix) if n == None else n_finder(matrix, n)
       
    coords = numpy_centroid(matrix)
    print(coords)
    center_x, center_y = coords['x'], coords['y']

    left_bound = center_x - n // 2 if (center_x - n // 2) >= 0 else 0
    right_bound = center_x + n // 2 if (center_x + n) < len(matrix[0]) else (len(matrix[0]) - 1)
    bottom_bound = center_y + n // 2 if (center_y + n) < len(matrix) else len(matrix) - 1
    top_bound = center_y - n // 2 if (center_y - n) >= 0 else 0

    cropped_img = img[top_bound:bottom_bound, left_bound:right_bound]
    cropped_path = path[0:(len(path) - 4)] + "_cropped" + extension
        
    cv.imwrite(cropped_path, cropped_img)
