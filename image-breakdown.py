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
    tl_x = 7
    tl_y = 0       # setting approximate pixel divisons/locations for crop function
    br_x = 92
    br_y = 87
    x_space = 85
    y_space = 87
    

    for i, seg in enumerate(box_vec):
      cropseg = seg.crop((tl_x,tl_y,br_x,br_y))  #crops current copy of each image and appends to new list
      crop_vec.append(cropseg)
      tl_x += x_space   
      br_x += x_space  
      if i==2 or i==5:
         tl_x = 7           #changing deliminters to iterate through whole sudoku image
         br_x = 92
         tl_y += y_space
         br_y += y_space
      

    return crop_vec


def digit_read(crop_vec):
   #Iterating through list of images and creating tensor of digits 
   unslvd_tensor = [[[0 for _ in range(9)] for _ in range(3)] for _ in range(3)]

   for i,seg in enumerate(crop_vec):
      #Converting image from PIL (png) to CV format
      crop_cv = cv2.cvtColor(np.array(seg), cv2.COLOR_RGB2BGR)
      #Preprocess image to gray scale, threshold it and invert it
      gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)
      thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

      #Run tesseract OCR to read the digits 
      digit_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
      text = pytesseract.image_to_string(thresh, config=digit_config).strip()

      #Read and store results
      if text.isdigit():
         print(int(text))
         print(i)
      else:
        print("failure to read\n")
         
      return crop_cv
      



crop_vec = nine_split(imgfile)
crop_cv = digit_read(crop_vec)

#for i,seg in enumerate(crop_vec):   #checking boxes are split correctly; may need to hit Ctrl+C afterward if your terminal freaks out
#   seg.show()






    

