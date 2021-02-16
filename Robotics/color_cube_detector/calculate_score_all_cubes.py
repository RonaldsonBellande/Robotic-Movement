from count_number_of_all_cubes import *
import numpy as np
from glob import glob
from os.path import basename
 
image_path = "images_data/"

file_data = np.genfromtxt(image_path + 'all_cube_file_data.csv', delimiter=',', dtype='int')

image_correct = dict([(item[0],(item[1], item[2], item[3])) for item in file_data])
 
cube_counted = 0
images = [count for count in glob(image_path +'*') if 'jpg' in count]

count_number_of_cubes_all_cubes = detect_cube()

for image in images:
    count = int(image[-6:-4])
    img = cv2.imread(image, -1)
    file_name = basename(image)
    
    nummber_of_yellow_cubes, number_of_green_cubes, number_of_red_cubes = count_number_of_cubes_all_cubes.count_cubes_in_picture(img, file_name)
    if nummber_of_yellow_cubes == image_correct[count][0] and number_of_green_cubes == image_correct[count][1] and number_of_red_cubes == image_correct[count][2]:
        cube_counted += 1
    
print("Number_of_cubes_detected: {}/{}".format(cube_counted, len(images)))
