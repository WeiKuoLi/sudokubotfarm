## Code to import scraped sudoku png file;
# splice into 9 individual boxes; 
# create 3x3x9 tensor of integers; 
# build model to fill integers and correctly solve puzzle 
from PIL import Image


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


crop_vec = nine_split(imgfile)

#for i,seg in enumerate(crop_vec):   #checking boxes are split correctly; may need to hit Ctrl+C afterward if your terminal freaks out
#   seg.show()
crop_vec[3].show()





    

