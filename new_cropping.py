#ipmorting libraries
import cv2 as cv
from PIL import Image
import numpy as np

#example coordinates of a square
#top left, bottom left, bottom right, top right
x = [20, 20, 40, 40]
y = [20, 40, 40, 20]

#the following is stack overflow code to make a matrix of a bunch of zeroes and
#a square of ones which we designated by the two lists above
contours = np.stack((x, y), axis = 1)
polygon = np.array([contours], dtype = np.int32)
zero_mask = np.zeros((100, 100), np.uint8)
polyMask = cv.fillPoly(zero_mask, polygon, 1)

#a dictionary of the start and end coordinates
coordinates = dict()
first_one = False

#iterating through the mask matrix
for i in range(len(polyMask)):
    
    for j in range(len(polyMask[i])):

        #if we've reached the first one, then we can keep track of it in the coordinates dictionary
        if polyMask[i][j] == 1 and first_one == False:
            coordinates['start'] = (i, j)
            first_one = True
        
        #keep overwriting the end entry to the coordinate dictionary everytime we see a 1; eventually we will
        #reach the lsat 1 in the mask
        if polyMask[i][j] == 1 and first_one == True:
            coordinates['end'] = (i, j)

#print(coordinates)


#print(polyMask)
#cv.imwrite("newimage.png", polyMask)

"""
The new code starts here onwards that would crop the images
"""
#given a 2D matrix of zeroes and ones (matrix) and a column index from 0-(Len(matrix) - 1) (col_index),
#returns the number of ones in the column, and the starting and ending index of the ones in the column
def num_ones_in_col(matrix, col_index):
    num_ones = 0
    first_one = FALSE
    start_col_index = 0
    end_col_index = 0
    #the integer i represents the number of rows
    #we're gonna go through the different rows of
    #column col_index to find the number of ones
    #in the column
    for i in range(len(matrix)):
        if(matrix[i][col_index] == 1 and first_one == FALSE):
            num_ones += 1
            start_col_index = i
            end_col_index = i #we include this line because of the edge case that there is only one number one in a column
            first_one = True
        if(matrix[i][col_index] == 1 and first_one == True):
            num_ones += 1
            end_col_index = i
    return num_ones, start_col_index, end_col_index

#given a 2D matrix of zeroes and ones (matrix) and a row index from 0-(Len(matrix) - 1) (row_index),
#returns the number of ones in the row, and the starting and ending index of the ones in the row
def num_ones_in_row(matrix, row_index):
    num_ones = 0
    first_one = FALSE
    start_row_index = 0
    end_row_index = 0
    #the integer j represents the number of columns
    #we're gonna go through the different columns
    #row row_index to find the number of ones
    #in the row
    for j in range(len(matrix[row_index])):
        if(matrix[row_index][j] == 1 and first_one == False):
            first_one == True
            start_row_index = j
            end_row_index = j # we include this just in case 
            num_ones += 1
        if(matrix[row_index][j] == 1 and first_one == True):
            end_row_index = j
            num_ones += 1
    return num_ones, start_row_index, end_row_index

#this function gives us the start and end indicies of the column with the most amount of ones
#for a 2d matrix (matrix) of ones and zeros
def loc_max_ones_col(matrix):
    max_num_ones = 0
    max_start_col_index = 0
    max_end_col_index = 0
    for i in range(len(matrix)):
        current, _, _ = num_ones_in_col(matrix, i)
        if(current > max_num_ones):
            max_num_ones, max_start_col_index, max_end_col_index = num_ones_in_col(matrix, i)
    return max_start_col_index, max_end_col_index

#this function gives us the start and end indicies of the row with the most amount of ones
#for a 2d matrix (matrix) of ones and zeros
def loc_max_ones_row(matrix):
    max_num_ones = 0
    max_start_row_index = 0
    max_end_row_index = 0
    for j in range(len(matrix[0])):
        current, _, _ = num_ones_in_row(matrix, j)
        if(current > max_num_ones):
            max_num_ones, max_start_row_index, max_end_row_index = num_ones_in_row(matrix, i)
    return max_start_row_index, max_end_row_index

#given a matrix of ones and zeros, finds the coordinates of the 
#center of ones in the matrix
def centroid_finder(matrix):
    row_start, row_end = loc_max_ones_row(matrix)
    col_start, col_end = loc_max_ones_col(matrix)

    centroid_x = (row_start + row_end) // 2
    centroid_y = (col_start + col_end) // 2

    return centroid_x, centroid_y

#given the filepath of an image (in .jpg) and its corresponding mask of ones and zeros,
#this function crops and stores the image in the same directory
#where n is half the length of the desired width and height
def cropper(img_path, matrix, n):

    img = cv.imread(img_path)

    center_x, center_y = centroid_finder(matrix)

    left_bound = center_x - n if (center_x - n) >= 0 else 0
    right_bound = center_x + n if (center_x + n) < len(matrix[0]) else (len(matrix[0]) - 1)
    bottom_bound = center_y + n if (center_y + n) < len(matrix) else len(matrix) - 1
    top_bound = center_y - n if (center_y - n) >= 0 else 0

    cropped_img = img[left_bound:right_bound, top_bound:bottom_bound]
    cropped_path = img_path[0:(len(img_path) - 4)] + "_cropped.jpg"

    cv.imwrite(cropped_path, cropped_img)


