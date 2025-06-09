## Code to import scraped sudoku png file;
# splice into 9 individual boxes; 
# create 3x3x9 tensor of integers; 
# build model to fill integers and correctly solve puzzle 
from PIL import Image
import cv2
import pytesseract
import numpy as np


imgfile = Image.open("tmp_image.png")

#first creating 9 copies of the image to individually crop

def nine_split(imgfile):
    
    box_vec = [imgfile.copy() for _ in range(9)] #creates a list of 9 copies of the image
    crop_vec = []                                #preallocates new list for cropped images
    tl_x = 0            #need to change depending on version of image (main or my branch)
    tl_y = 0       # setting approximate pixel divisons/locations for crop function
    br_x = 87
    br_y = 87
    x_space = 87
    y_space = 87
    

    for i, seg in enumerate(box_vec):
      cropseg = seg.crop((tl_x,tl_y,br_x,br_y))  #crops current copy of each image and appends to new list
      crop_vec.append(cropseg)
      tl_x += x_space   
      br_x += x_space  
      if i==2 or i==5:
         tl_x = 0           #changing deliminters to iterate through whole sudoku image
         br_x = 87
         tl_y += y_space
         br_y += y_space
      

    return crop_vec


def digit_read(crop_vec):
   #Iterating through list of images and creating tensor of digits 
   unslvd_boxes = [[[0 for _ in range(9)] for _ in range(3)] for _ in range(3)]

   subreg_size = 29
   

   for i,seg in enumerate(crop_vec):
        
        #Converting image from PIL (png) to CV format
        crop_cv = cv2.cvtColor(np.array(seg), cv2.COLOR_RGB2BGR)
        


        #Preprocess image to gray scale, threshold it and invert it
        gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        digit_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'

        #Run tesseract OCR to read the digits 
        
        for a in range(3):  # Rows
            for b in range(3):  # Columns
                # Crop 29x29 sub-region
                x1 = b * subreg_size  # 0, 29, 58
                y1 = a * subreg_size  # 0, 29, 58
                x2 = x1 + subreg_size  # 29, 58, 87
                y2 = y1 + subreg_size  # 29, 58, 87
                sub_region = thresh[y1:y2, x1:x2]
                
                
                
                
                # Run OCR on sub-region
                text = pytesseract.image_to_string(sub_region, config=digit_config).strip()
                if text.isdigit():
                    #print(int(text))
                    unslvd_boxes[a][b][i] = int(text)
                

   return  unslvd_boxes
    
    
def puzzle_reconstruct(unslvd_boxes):
    reconpuz = np.zeros((9,9), dtype=int)
    np_unslvd = np.array(unslvd_boxes)
    for x in range((np_unslvd.shape[2])):
        for y in range((np_unslvd.shape[1])):
            for z in range((np_unslvd.shape[0])):
                #print(f"{x},{y},{z}")
                if x <= 2:
                    reconpuz[y][z+(x*3)] = np_unslvd[y][z][x]
                elif x > 2 and x <= 5:
                    reconpuz[y+3][z+(x*3-9)] = np_unslvd[y][z][x]
                elif x > 5:
                    reconpuz[y+6][z+(x*3-18)] = np_unslvd[y][z][x]

    return reconpuz


crop_vec = nine_split(imgfile)
unslvd_boxes = digit_read(crop_vec)     #currently misses 5 numbers... will work on and maybe model can fill them in
reconpuz = puzzle_reconstruct(unslvd_boxes)

