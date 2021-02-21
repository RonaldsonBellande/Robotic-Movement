import numpy as np
import cv2
from glob import glob
import os
from os.path import basename
import matplotlib.pyplot as plt

def save_corner_image(img, file_name, corner_detector):
    image_path = "images_data/"
    image_number = [count for count in glob(image_path+'*') if 'jpg' in count]
        
    if corner_detector == 'corner_harris':
        image_output = "corner_harris/base/"
    elif corner_detector == 'corner_harris_with_subpixel':
        image_output = "corner_harris/subpixel/"
    else:
        image_output = "good_features_to_track/"

    for i in range(len(image_number)):
        cv2.imwrite(os.path.join(image_output, str(file_name)), img)
        cv2.waitKey(0)
        
def save_personal_kernel(img, file_name, kernel):
    image_path = "images_data/"
    image_number = [count for count in glob(image_path+'*') if 'jpg' in count]
        
    if kernel == 'kernel_1':
        image_output = "personel_kernel/kernel_1"
    else:
        image_output = "personel_kernel/kernel_2"

    for i in range(len(image_number)):
        cv2.imwrite(os.path.join(image_output, str(file_name)), img)
        cv2.waitKey(0)

def save_edge_image(img, file_name, edge_detector):
    image_path = "images_data/"
    image_number = [count for count in glob(image_path+'*') if 'jpg' in count]
        
    if edge_detector == 'laplacian':
        image_output = "laplacian_edge_detector/base/"
    elif edge_detector == 'sobel_x_axis':
        image_output = "sobel_edge_detector/x_axis/base/"
    elif edge_detector == 'sobel_y_axis':
        image_output = "sobel_edge_detector/y_axis/base/"
    elif edge_detector == 'merge_xy_axis':
        image_output = "sobel_edge_detector/merge_xy_axis/base/"
    elif edge_detector == 'canny':
        image_output = "canny_edge_detector/base/"
    elif edge_detector == 'laplacian_plus_picture':
        image_output = "laplacian_edge_detector/plus_image/"
    elif edge_detector == 'sobel_x_axis_plus_picture':
        image_output = "sobel_edge_detector/x_axis/plus_image/"
    elif edge_detector == 'sobel_y_axis_plus_picture':
        image_output = "sobel_edge_detector/y_axis/plus_image/"
    elif edge_detector == 'merge_xy_axis_plus_picture':
        image_output = "sobel_edge_detector/merge_xy_axis/plus_image/"
    else:
        image_output = "canny_edge_detector/plus_image/"

    for i in range(len(image_number)):
        cv2.imwrite(os.path.join(image_output, str(file_name)), img)
        cv2.waitKey(0)
        
image_path = "images_data/"
images = [count for count in glob(image_path +'*') if 'jpg' in count]

for image in images:
    img = cv2.imread(image, -1)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (B, G, R) = cv2.split(img)
    file_name = basename(image)
    
    image_noise_removal = cv2.GaussianBlur(gray_scale,(3,3),0)
    
    laplacian = cv2.Laplacian(gray_scale, cv2.CV_64F)
    save_edge_image(laplacian, file_name, edge_detector = "laplacian")
    
    sobel_x_axis = cv2.Sobel(image_noise_removal, cv2.CV_64F,1,0,ksize=5)
    save_edge_image(sobel_x_axis, file_name, edge_detector = "sobel_x_axis")
    
    sobel_y_axis = cv2.Sobel(image_noise_removal, cv2.CV_64F,0,1,ksize=5)
    save_edge_image(sobel_y_axis, file_name, edge_detector = "sobel_y_axis")
    
    merge_xy_axis = sobel_x_axis + sobel_y_axis
    save_edge_image(sobel_y_axis, file_name, edge_detector = "merge_xy_axis")
    
    canny = cv2.Canny(gray_scale, 100,200)
    save_edge_image(canny, file_name, edge_detector = "canny")
    
    ## Starts Here the combination of the images
    
    #contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #laplacian_plus_picture = cv2.drawContours(img, contours, 0, (0,255,0), 2)
    #save_edge_image(laplacian_plus_picture, file_name, edge_detector = "laplacian_plus_picture")
    
    #contours, hierarchy = cv2.findContours(sobel_x_axis, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #sobel_x_axis_plus_picture = cv2.drawContours(img, contours, 0, (0,255,0), 2)
    #save_edge_image(sobel_x_axis_plus_picture, file_name, edge_detector = "sobel_x_axis_plus_picture")
    
    #contours, hierarchy = cv2.findContours(sobel_y_axis, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #sobel_y_axis_plus_picture = cv2.drawContours(img, contours, 0, (0,255,0), 2)
    #save_edge_image(sobel_y_axis_plus_picture, file_name, edge_detector = "sobel_y_axis_plus_picture")
    
    #contours, hierarchy = cv2.findContours(merge_xy_axis, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #merge_xy_axis_plus_picture = cv2.drawContours(img, contours, 0, (0,255,0), 2)
    #save_edge_image(merge_xy_axis_plus_picture, file_name, edge_detector = "merge_xy_axis_plus_picture")
    
    kernel = np.array([[-1, 2, -1],
                       [2, -1, 2],
                       [-1, 2, -1]])
    
    img_my_kernel = cv2.filter2D(gray_scale, -1, kernel)
    save_personal_kernel(img_my_kernel, file_name, kernel = "kernel_1")
    
    kernel = np.array([[11, 2, -1],
                       [2, 0, 2],
                       [1, 2, -1]])
    
    img_my_kernel = cv2.filter2D(gray_scale, -1, kernel)
    save_personal_kernel(img_my_kernel, file_name, kernel = "kernel_2")
    
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    canny_plus_picture = cv2.drawContours(img, contours, -1, (255,0,0), 3)
    save_edge_image(canny_plus_picture, file_name, edge_detector = "canny_plus_picture")
    
    
    corner_harris = cv2.cornerHarris(image_noise_removal,2,3,0.04)
    corner_harris = cv2.dilate(corner_harris, None)
    img[corner_harris>0.01*corner_harris.max()]=[0,0,255]
    save_corner_image(img, file_name, corner_detector = "corner_harris")
    
    
    ret, corner_harris_with_subpixel = cv2.threshold(corner_harris,0.01*corner_harris.max(),255,0)
    corner_harris_with_subpixel = np.uint8(corner_harris_with_subpixel)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corner_harris_with_subpixel)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image_noise_removal,np.float32(centroids),(5,5),(-1,-1),criteria)
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,1],res[:,1]] = [0,255,0]
    save_corner_image(img, file_name, corner_detector = "corner_harris_with_subpixel")
    
    
    good_features_to_track = cv2.goodFeaturesToTrack(image_noise_removal,25,0.01,10)
    good_features_to_track = np.int0(good_features_to_track)
    
    for i in good_features_to_track:
        x_axis, y_axis = i.ravel()
        cv2.circle(img,(x_axis, y_axis),3,255,-1)
    
    save_corner_image(img, file_name, corner_detector = "good_features_to_track")
    



