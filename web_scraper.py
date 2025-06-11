from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
from os import remove
from os.path import join, dirname
import time

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

def crop_and_downsample_image(filepath):
    '''
    Opens the image at filepath, crops it to shape, and downsamples it to easily export for image inference.

    Args:
        filepath (str): the path of the image to crop and downsample.

    Returns:
        str: the path of the cropped and downsampled image. If the original image could not be opened, return the empty string
    '''
    # Set up cropping parameters
    crop_box = (100, 126, 526, 552) # left, top, right, bottom
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
    raise NotImplementedError('Image submission cannot be implemented until the server is known.')

if __name__ == '__main__':
    file = extract_screenshot('https://sudoku.com/expert/')
    updated_file = crop_and_downsample_image(file)
    #response = submit_image_for_inference(updated_file)

