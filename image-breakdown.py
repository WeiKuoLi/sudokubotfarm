## Code to import scraped sudoku png file;
# splice into 9 individual boxes; 
# create 9 3x3 matrices of integers; 
# build model to fill integers and correctly solve puzzle 
from PIL import Image


#-------------------------------------------------------
# Code to import and splice the png image:
imgfile = Image.open("tmp_image.png")

def nine_split(imgfile):
    #first creating 9 copies of the image to individually crop
    box_vec = [imgfile.copy() for _ in range(9)] #creates a list of 9 copies of the image
    crop_vec = []
    tl_x = 7
    tl_y = 0
    br_x = 92
    br_y = 87
    x_space = 85
    y_space = 87
    
    for i, seg in enumerate(box_vec):
      cropseg = seg.crop((tl_x,tl_y,br_x,br_y))  
      crop_vec.append(cropseg)
      tl_x += x_space
      br_x += x_space  
      if i==2 or i==5:
         tl_x = 7
         br_x = 92
         tl_y += y_space
         br_y += y_space
      
      
      

    return crop_vec
    
crop_vec = nine_split(imgfile)

for i,seg in enumerate(crop_vec):
   seg.show()
#crop_vec[5].show()



    

