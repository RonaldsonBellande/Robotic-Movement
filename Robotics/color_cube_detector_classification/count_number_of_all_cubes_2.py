import cv2
import numpy as np
from glob import glob
import os
from scipy import ndimage

class detect_cube(object):
    
    def __init__(self):
        
        # Lower Mask red
        self.red_lower = np.array([0,10,0])
        self.red_upper = np.array([5,255,255])
        
        # Upper Mask red
        #self.red_lower = np.array([79, 56, 56])
        #self.red_upper = np.array([180,255,255])
        
        self.yellow_lower = np.array([8,167,141])
        self.yellow_upper = np.array([41,235,255])

        self.green_lower = np.array([18,14,4])
        self.green_upper = np.array([64,189,100])
        
        
    def count_cubes_in_picture(self, img, file_name):
        
        mask_yellow = self.filter_image(img, self.yellow_lower, self.yellow_upper)
        self.save_mask(mask_yellow, file_name, color = 'yellow')
        
        num_yellow = self.detect_blob(mask_yellow, file_name, color = 'yellow')
        number_yellow = num_yellow
        self.add_mask_and_image(number_yellow, img, mask_yellow, file_name, color = 'yellow')

        mask_green = self.filter_image(img, self.green_lower, self.green_upper)
        self.save_mask(mask_green, file_name, color = 'green')
        
        num_green = self.detect_blob(mask_green, file_name, color = 'green')
        number_green = num_green
        self.add_mask_and_image(number_green, img, mask_green, file_name, color = 'green')
        
        mask_red = self.filter_image(img, self.red_lower, self.red_upper)
        self.save_mask(mask_red, file_name, color = 'red')
        
        num_red = self.detect_blob(mask_red, file_name, color = 'red')
        number_red = num_red
        self.add_mask_and_image(number_red, img, mask_red, file_name, color = 'red')

        return num_yellow, num_green, num_red
    
    
    def detect_blob(self, mask, file_name, color):

        mask = mask
        mask = cv2.GaussianBlur(mask,(5,5),0)
        
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0;
        params.maxThreshold = 350;
        

        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 10000

        #params.filterByColor = True
        params.filterByCircularity = False
        
        params.filterByConvexity = False

        # builds a blob detector with the given parameters 
        detector = cv2.SimpleBlobDetector_create(params)

        # use the detector to detect blobs.
        keypoints = detector.detect(mask)
        
        image_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.save_blob_detector_image(image_with_keypoints, file_name, color)
        
        return len(keypoints)
    
    
    def filter_image(self, img, hsv_lower, hsv_upper):

        img3 = cv2.GaussianBlur(img,(5,5),0)
        img1 = cv2.medianBlur(img3, 11)
        cv2.addWeighted(img,1.5,img,-0.5,0,img1)

        #Convert BGR to HSV for the image 
        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        
        # Modify mask
        mask = cv2.inRange(hsv,hsv_lower,hsv_upper)
        
        return mask
    
    def save_mask(self, img, file_name, color):
        
        image_path = "images_data/"
        image_number = [count for count in glob(image_path+'*') if 'jpg' in count]
        
        if color == 'yellow':
            image_output = "yellow_mask_opposite/"
        elif color == 'green':
            image_output = "green_mask_opposite/"
        else:
            image_output = "red_mask_opposite/"

        for i in range(len(image_number)):
            cv2.imwrite(os.path.join(image_output, str(file_name)), img)
            cv2.waitKey(0)
            
    
    def save_blob_detector_image(self, mask, file_name, color):
        
        image_path = "images_data/"
        image_number = [count for count in glob(image_path+'*') if 'jpg' in count]
        
        if color == 'yellow':
            image_output = "yellow_blob_detection/"
        elif color == 'green':
            image_output = "green_blob_detection/"
        else:
            image_output = "red_blob_detection/"

        for i in range(len(image_number)):
            cv2.imwrite(os.path.join(image_output, str(file_name)), mask)
            cv2.waitKey(0)
    
    def add_mask_and_image(self, number, img, mask, file_name, color):
        
        image_path = "images_data/"
        image_number = [count for count in glob(image_path+'*') if 'jpg' in count]
        
        #mask_to_color = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        
        #roi_corners = np.array([[(10,10), (150,150), (10,150)]], dtype=np.int32)
        #channel_count = img.shape[2]
        #ignore_mask_color = (255,)*channel_count
        #cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        
        
        #mask[200:500,200:500] = 255
        #rect = cv2.boundingRect(mask)
        
        #if color == 'yellow':
            #img = np.ones([600,600,3], dtype=np.uint8)
            #img = img[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
        
        # More complex
        #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        #mask_out = cv2.subtract(mask,img)
        #combine_image = cv2.subtract(mask,mask_out)
        
        # Simple
        combine_image = cv2.bitwise_and(img, img, mask = mask)
        combine_image[combine_image == 0] = 255
        
        # Crop image from white space
        gray = cv2.cvtColor(combine_image, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
        
        contours = cv2.findContours(morphed,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        count = 0
        if len(contours) != 0:
            count = 0
            for i in range(len(contours)):
                cnt = sorted(contours, key=cv2.contourArea, reverse=True)[-1]
                
                # The amount of object found
                if cv2.contourArea(cnt) > 2:
                    count +=1
                    
                    if count == 1:
                        x,y,w,h = cv2.boundingRect(cnt)
                        combine_image = combine_image[y:y+h, x:x+w]
                    elif count == 2:
                        x_2,y_2,w_2,h_2 = cv2.boundingRect(cnt)
                        combine_image = combine_image[y_2:y_2+h_2, x_2:x_2+w_2]
                
                else:
                    combine_image = cv2.bitwise_and(img, img, mask = mask)
                    combine_image[combine_image == 0] = 255
                    
        else:
            combine_image = cv2.bitwise_and(img, img, mask = mask)
            combine_image[combine_image == 0] = 255
            
        if combine_image == []:
            combine_image = cv2.bitwise_and(img, img, mask = mask)
            combine_image[combine_image == 0] = 255
    
    
        if color == 'yellow':
            image_output = "create_model_from_cubes_2/yellow_cubes/"
        elif color == 'green':
            image_output = "create_model_from_cubes_2/green_cubes/"
        else:
            image_output = "create_model_from_cubes_2/red_cubes/"
        
        if number != 0:
            if count == 1 or count == 0:
                
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, str(file_name)), combine_image)
                
                image_rotate = ndimage.rotate(combine_image, 60)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_1" + str(file_name)), image_rotate)
            
                image_rotate = ndimage.rotate(combine_image, 120)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_2" + str(file_name)), image_rotate)
            
                image_rotate = ndimage.rotate(combine_image, 180)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_3" + str(file_name)), image_rotate)
            
                image_rotate = ndimage.rotate(combine_image, 240)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_4" + str(file_name)), image_rotate)
            
                image_rotate = ndimage.rotate(combine_image, 300)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_5" + str(file_name)), image_rotate)
                    cv2.waitKey(0)
            
            elif count == 2:
                
                combine_image = cv2.bitwise_and(img, img, mask = mask)
                combine_image[combine_image == 0] = 255
                
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "normal_2" + str(file_name)), combine_image)
                
                image_rotate = ndimage.rotate(combine_image, 60)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_1_2" + str(file_name)), image_rotate)
            
                image_rotate = ndimage.rotate(combine_image, 120)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_2_2" + str(file_name)), image_rotate)
            
                image_rotate = ndimage.rotate(combine_image, 180)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_3_2" + str(file_name)), image_rotate)
            
                image_rotate = ndimage.rotate(combine_image, 240)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_4_2" + str(file_name)), image_rotate)
            
                image_rotate = ndimage.rotate(combine_image, 300)
                image_rotate[image_rotate == 0] = 255
                for i in range(len(image_number)):
                    cv2.imwrite(os.path.join(image_output, "rotate_5_2" + str(file_name)), image_rotate)
                    cv2.waitKey(0)
                        
    
