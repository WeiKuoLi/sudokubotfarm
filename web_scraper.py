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
        filepath (str): the path to the screenshot
    '''

    # construct save location (in this directory)
    filepath = join(dirname(__file__),'tmp_image.png')

    # Set up selenium browser
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1280, 820)
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

def crop_image(filepath):
    # Set up cropping parameters
    crop_box = (100, 126, 525, 551) # left, top, right, bottom
    new_size = (262,262) # downsample size. 262x262 corresponds to 9- 28x28 cells + 10- 1px borders (doesn't quite match, but close enough)

    # Wrap in try/except to make sure we get `something`
    try:
        img = Image.open(filepath)
    except:
        print(f'Could not open file {filepath}')
        return False

    cropped = img.crop(crop_box)
    downsampled = cropped.resize(new_size, resample=Image.LANCZOS)
    img.close()
    cropped.close()
    downsampled.save(filepath)
    return filepath

if __name__ == '__main__':
    file = extract_screenshot('https://www.sudoku.com/expert/')
    updated_file = crop_image(file)