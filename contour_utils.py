import numpy as np 
import cv2 
import os, sys 
import matplotlib.pyplot as plt

def file2poly(im_loc):
    '''
    This function loads an image from disk, runs a contour analysis on it and returns the 
    list of closed contours as a json 
    
    Parameters 
    ----------
    im_loc: location of the image (.png, .jpg etc) 

    Returns
    -------
    a dictionary containing keys as polygons and 2 lists per key (x and y locations) 
    '''
    
    assert os.path.isfile(im_loc), "im_loc {} is not a regular file".format(im_loc)
    im = cv2.imread(im_loc, cv2.IMREAD_GRAYSCALE)
    ret, im_thresh = cv2.threshold(im, 127, 255, 0)
    im_thresh = 255 - im_thresh
    contour_dict  = im2poly(im_thresh) 
    return contour_dict

def clean_contours(contours):
    '''
    This function cleans the contours list. 
    1. Remove small contours 
    2. Remove contours with just one point 
    '''
    # import ipdb; ipdb.set_trace()
    contours_clean = [] 
    im_area = 224*224
    for cont in contours:
        if cv2.contourArea(cont) > 0.01*im_area and len(cont)>1:
            contours_clean.append(cont)
    return contours_clean

def draw_contours(contours):
    '''
    This function draws the contours on a blank array 
    '''
    image = np.zeros((224,224,3)).astype(np.uint8)
    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    plt.imshow(image)
    plt.show()

def im2poly(im):
    '''
    Refer to doc string in file2poly. 
    This is a helper function that finds contours of images and returns them as a dictionary
    '''
    im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = clean_contours(contours)
    # draw_contours(contours)
    contour_dict = {}
    for idx, contour in enumerate(contours):
        contour_dict[str(idx)] = {}
        contour_dict[str(idx)]["x"] = [] 
        contour_dict[str(idx)]["y"] = []
        for pt in contour:
            contour_dict[str(idx)]["x"].append(int(pt[0][0]))
            contour_dict[str(idx)]["y"].append(int(pt[0][1]))
        contour_dict[str(idx)]["x"] = tuple(contour_dict[str(idx)]["x"])
        contour_dict[str(idx)]["y"] = tuple(contour_dict[str(idx)]["y"])
    
    return contour_dict

