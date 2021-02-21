import cv2
import numpy as np
from glob import glob
import os

class detect_cube(object):
    
    def __init__(self):
        
        #self.yellow_lower = np.array([8, 169, 131])
        #self.yellow_upper = np.array([41, 235, 255])
        
        self.yellow_lower = np.array([8, 167, 141])
        self.yellow_upper = np.array([41, 235, 255])

        self.green_lower = np.array([18, 14, 4])
        self.green_upper = np.array([64,189,100])
        
        
    def count_cubes_in_picture(self, img, file_name):
        
        mask_yellow = self.filter_image(img, self.yellow_lower, self.yellow_upper)
        self.save_mask(mask_yellow, file_name, color = 'yellow')
        
        num_yellow = self.detect_blob(mask_yellow, file_name, color = 'yellow')

        mask_green = self.filter_image(img, self.green_lower, self.green_upper)
        self.save_mask(mask_green, file_name, color = 'green')
        
        num_green = self.detect_blob(mask_green, file_name, color = 'green')

        return num_yellow, num_green
    
    
    def detect_blob(self, mask, file_name, color):

        mask = cv2.GaussianBlur(mask,(5,5),0)
        #img1 = cv2.medianBlur(img1, 5)
        
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0;
        params.maxThreshold = 350;
        

        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 10000

        #params.filterByColor = True
        params.filterByCircularity = False
        
        #Filter Inertia
        #params.filterByInertia = True

        params.filterByConvexity = False

        # builds a blob detector with the given parameters 
        detector = cv2.SimpleBlobDetector_create(params)

        # use the detector to detect blobs.
        keypoints = detector.detect(mask)
        
        image_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.save_blob_detector_image(image_with_keypoints, file_name, color)
        
        return len(keypoints)
    
    
    
    def filter_image(self, img, hsv_lower, hsv_upper):

        #pixel = 4
        #temp = np.ones((pixel,pixel),np.float32)/pow(pixel,2)

        img3 = cv2.GaussianBlur(img,(5,5),0)
        img1 = cv2.medianBlur(img3, 11)
        cv2.addWeighted(img,1.5,img,-0.5,0,img1)

        #Convert BGR to HSV for the image 
        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        
        # Modify mask
        mask = cv2.inRange(hsv,hsv_lower,hsv_upper)
        
        #mark = cv2.inRange(temp,hsv_lower,hsv_upper)
        #mark = cv2.dilate(mark, None, iterations=1)
        #cv2.addWeighted(mask,1.5,mask,-0.5,0,mask)
        
        # Making it so that it shows the detection object as black and everything else as white
        mask = ~mask
        
        return mask
    
    def save_mask(self, img, file_name, color):
        
        image_path = "images_data/"
        image_number = [count for count in glob(image_path+'*') if 'jpg' in count]
        
        if color == 'yellow':
            image_output = "yellow_mask/"
        else:
            image_output = "green_mask/"

        for i in range(len(image_number)):
            cv2.imwrite(os.path.join(image_output, str(file_name)), img)
            cv2.waitKey(0)
            
    
    def save_blob_detector_image(self, mask, file_name, color):
        
        image_path = "images_data/"
        image_number = [count for count in glob(image_path+'*') if 'jpg' in count]
        
        if color == 'yellow':
            image_output = "yellow_blob_detection/"
        else:
            image_output = "green_blob_detection/"

        for i in range(len(image_number)):
            cv2.imwrite(os.path.join(image_output, str(file_name)), mask)
            cv2.waitKey(0)
    
    
