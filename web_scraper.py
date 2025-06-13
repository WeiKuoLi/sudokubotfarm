from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
from os import remove
from os.path import join, dirname
import time
from image_breakdown import extract_puzzle, solve_puzzle, print_puzzle

def clean_up_images(filepath:str)->bool:
    '''
    Attempts to remove the file at the given filepath.

    Args:
        filepath (str): Path to the file to be removed.

    Returns:
        bool: True if the file was successfully removed, False otherwise.
    '''
    # Wrap in try/except to make sure we don't crash
    try:
        remove(filepath) # os.remove()
        return True
    except:
        print(f'Could not remove file {filepath}')
        return False

def extract_screenshot(web_url:str)->str:
    '''
    Opens the site provided in web_url and takes a screenshot.

    Args:
        web_url (str): the URL of the website to be screenshotted.
        
    Returns:
        str: the path to the screenshot
    '''

    # construct save location (in this directory)
    filepath = join(dirname(__file__),'tmp_image.png')

    # Set up selenium browser
    options = Options()
    options.headless = True
    options.add_argument("--headless=new")  # more reliable in newer Chrome versions
    options.add_argument("--disable-gpu")   # optional but common
    options.add_argument("--window-size=1280,820")
    driver = webdriver.Chrome(options=options)
    #driver.set_window_size(1280, 820)
    driver.get(web_url)

    # Sudoku.com has an annoying "hint" at the start. Use the mouse to bypass that.
    actions = ActionChains(driver)
    actions.move_by_offset(300, 320).click().perform() # move to 300, 320 and click mouse
    time.sleep(1) # purge any screen artifacts

    # Now the site is primed. Take a screenshot and store it to file.
    driver.get_screenshot_as_file(filepath)

    # Close the browser and return
    driver.quit()
    return filepath

def sudoku_bot(web_url:str)->bool:
    '''
    Opens the site provided in web_url and plays Sudoku.com

    Args:
        web_url (str): the URL of the website to be played.
        
    Returns:
        bool: True if the sudoku was successfully solved, False otherwise.
    '''

    # construct save location (in this directory)
    filepath = join(dirname(__file__),'tmp_image.png')

    # Set up selenium browser
    options = Options()
    options.add_argument("--headless=new")  # more reliable in newer Chrome versions
    options.add_argument("--disable-gpu")   # optional but common
    options.add_argument("--window-size=1280,820")
    driver = webdriver.Chrome(options=options)
    #driver.set_window_size(1280, 820)
    driver.get(web_url)

    from selenium.webdriver.common.by import By
    element = driver.find_element(By.CSS_SELECTOR, "#game > canvas")
    driver.execute_script("arguments[0].style.border='2px solid red'", element)
    size = element.size
    
    # Sudoku.com has an annoying "hint" at the start. Use the mouse to bypass that.
    actions = ActionChains(driver)
    #actions.move_by_offset(300, 320).click().perform() # move to 300, 320 and cplick mouse
    actions.move_to_element_with_offset(element, 10, 10).click().perform()
    time.sleep(1) # purge any screen artifacts

    # Now the site is primed. Take a screenshot and store it to file.
    driver.get_screenshot_as_file(filepath)
  

    updated_file = crop_and_downsample_image(filepath)
    sudoku = submit_image_for_inference(updated_file)
    print_puzzle(sudoku)
    answer = solve_puzzle(sudoku)
    if answer is None:
        print('No solution found')
        return False
    print_puzzle(answer)

    
    
    box_l, box_t, box_r, box_b = 0,0, size['width'], size['height'] # sudoku puzzle box coordinates
    dx, dy = (box_r - box_l)//9, (box_b - box_t)//9 # size of each cell
    for i in range(9):
        for j in range(9):
            cell_answer = str(answer[i][j])
            if len(cell_answer)!=1 or not (str(answer[i][j]) in '123456789'):
                print(f'Invalid answer at {i},{j}')
                return False
            offset_x, offset_y = int((j+0.5)*dx), int((i+0.5)*dy) # coordinates of the cell
            if sudoku[i][j] == 0:
                print(f'Entering {cell_answer} at {offset_x}, {offset_y}')
                actions = ActionChains(driver)
                actions.move_to_element_with_offset(element, offset_x, offset_y).click().perform()
                time.sleep(0.1)
                actions.send_keys(cell_answer).perform()
                time.sleep(0.1)
                _tmp_filename = join(dirname(__file__),f"tmp/cell_{i}{j}.png")
                actions = ActionChains(driver)
                driver.get_screenshot_as_file(_tmp_filename)
    time.sleep(1)
    driver.get_screenshot_as_file("result.png")
    # Close the browser and return
    driver.quit()
    return  True

def crop_and_downsample_image(filepath):
    '''
    Opens the image at filepath, crops it to shape, and downsamples it to easily export for image inference.

    Args:
        filepath (str): the path of the image to crop and downsample.

    Returns:
        str: the path of the cropped and downsampled image. If the original image could not be opened, return the empty string
    '''
    # Set up cropping parameters
    #crop_box = (100, 126, 526, 552) # left, top, right, bottom
    crop_box = (108, 126, 534, 552) # left, top, right, bottom
    new_size = (262,262) # downsample size. 262x262 corresponds to 9- 28x28 cells + 10- 1px borders (doesn't quite match with the rasterizing, but close enough)

    # Wrap in try/except to make sure we get `something`
    try:
        img = Image.open(filepath)
    except:
        print(f'Could not open file {filepath}')
        return ''

    # Crop, downsample image
    cropped = img.crop(crop_box)
    downsampled = cropped.resize(new_size, resample=Image.LANCZOS)

    # clean up opened files
    img.close()
    cropped.close()

    # save to filepath and return
    downsampled.save(filepath)
    return filepath

def submit_image_for_inference(filepath):
    '''
    Submit the image to the appropriate webserver for inference

    Args:
        filepath (str): the path of the image to submit for inference

    Returns:
        List[List[int]]: the two-dimensional array of integers representing the starting board
    '''
    img = Image.open(filepath)
    puzzle = extract_puzzle(img)
    img.close()
    return puzzle

if __name__ == '__main__':
    for i in range(30):
        try:
            sudoku_bot('https://sudoku.com/easy/')
        except:
            print('Failed to solve sudoku')
    '''
    file = extract_screenshot('https://sudoku.com/easy/')
    updated_file = crop_and_downsample_image(file)
    response = submit_image_for_inference(updated_file)
    answer = solve_puzzle(response)
    print_puzzle(response)
    if answer is not None:
        print_puzzle(answer)
    else:
        print('No solution found')
    '''
