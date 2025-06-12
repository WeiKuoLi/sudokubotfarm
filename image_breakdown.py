## Code to import scraped sudoku png file;
# splice into 9 individual boxes; 
# create 3x3x9 tensor of integers; 
# build model to fill integers and correctly solve puzzle 

from PIL import Image
import cv2
import pytesseract
import numpy as np
import random


imgfile = Image.open("tmp_image.png")

#first creating 9 copies of the image to individually crop

def nine_split(imgfile):
    
    box_vec = [imgfile.copy() for _ in range(9)] #creates a list of 9 copies of the image
    crop_vec = []                                #preallocates new list for cropped images
    tl_x = 7            #need to change depending on version of image (main or my branch)
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
         br_x = 92          #resetting based on these delimieters
         tl_y += y_space
         br_y += y_space
      

    return crop_vec


def digit_read(crop_vec):
   #Iterating through list of images and creating tensor of digits 
   unslvd_boxes = np.zeros((3,3,9), dtype=int)

   subreg_size = 29
   

   for i,seg in enumerate(crop_vec):
        
        #Converting image from PIL (png) to CV format
        crop_cv = cv2.cvtColor(np.array(seg), cv2.COLOR_RGB2BGR)
        


        #Preprocess image to gray scale, threshold it and invert it
        gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        #thresh = cv2.threshold(gray, 0, 255,  cv2.THRESH_OTSU)[1]
        digit_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'

        #Run tesseract OCR to read the digits 
        
        for a in range(3):  # Rows
            for b in range(3):  # Columns
                # Crop 29x29 sub-region
                x1 = b * subreg_size  # 0, 29, 58
                y1 = a * subreg_size  # 0, 29, 58
                x2 = x1 + subreg_size  
                y2 = y1 + subreg_size  
                sub_region = thresh[y1:y2, x1:x2]
                
                
                
                
                # Run OCR on sub-region
                text = pytesseract.image_to_string(sub_region, config=digit_config).strip()
                if text.isdigit():
                    #print(int(text))
                    unslvd_boxes[a][b][i] = int(text)
    
    
                

   return  unslvd_boxes
    
    
def puzzle_reconstruct(unslvd_boxes):
    reconpuz = np.zeros((9,9), dtype=int)       #conversion to numpy array
    np_unslvd = np.array(unslvd_boxes)
    for x in range((np_unslvd.shape[2])):
        for y in range((np_unslvd.shape[1])):
            for z in range((np_unslvd.shape[0])):
                
                if x <= 2:
                    reconpuz[y][z+(x*3)] = np_unslvd[y][z][x]       #scrappy code but pieces back together the tensor into 
                elif x > 2 and x <= 5:                              #full empty puzzle
                    reconpuz[y+3][z+(x*3-9)] = np_unslvd[y][z][x]
                elif x > 5:
                    reconpuz[y+6][z+(x*3-18)] = np_unslvd[y][z][x]

    return reconpuz

def validity_test(puzz, row, col, num):
    #check if the number repeats in the row
    if num in puzz[row,:]:
        return False
    
    #Check if number repeats in column
    if num in puzz[:,col]:
        return False
    
    #check if number repeats in 3x3 box
    start_row, start_col = 3*(row//3), 3*(col//3)
    if num in puzz[start_row:start_row+3, start_col:start_col+3]:
        return False

    #Passes all tests 
    return True

def solution_seek(puzz):
    #start with empty cell 
    for row in range(puzz.shape[1]):
        for col in range(puzz.shape[1]):
            if puzz[row,col] ==0:
                #Try numbers 1-9
                for num in range(1,10):
                    if validity_test(puzz,row,col,num):
                        puzz[row,col] = num #Places number
                        if solution_seek(puzz):     #Uses recursion
                            return True
                        puzz[row,col] = 0   #if test didnt work set value back to zero and start over    
                return False
    return True #puzzle solved 

def extract_puzzle(imgfile):
    crop_vec = nine_split(imgfile)
    unslvd_boxes = digit_read(crop_vec)     
    reconpuz = puzzle_reconstruct(unslvd_boxes)
    return reconpuz.tolist()

def solve_puzzle(puzzle):
    puzzsolve = np.array(puzzle.copy(), dtype=int)     #copy so as to not change any aspects of the original
    if solution_seek(puzzsolve):
        return puzzsolve.tolist()
    else:
        return None

def print_puzzle(puzzle):
    for row in puzzle:
        _vis = [i if i != 0 else "." for i in row]
        print(*_vis)

if __name__ == "__main__":
    puzzle = extract_puzzle(imgfile)
    print("Problem: \n")
    print_puzzle(puzzle)

    answer = solve_puzzle(puzzle)
    if answer is not None:
        print("Solution: \n")
        print_puzzle(answer)
    else:
        print("Unsolvable")

