from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
from os import remove
from os.path import join, dirname

def clean_up_images(filepath):
    try:
        remove(filepath)
    except:
        print(f'Could not remove file {filepath}')

def extract_screenshot():
    filepath = join(dirname(__file__),'tmp_image.png')
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    driver.get('https://sudoku.com/expert')
    # content = driver.page_source.encode('unicode_escape').decode('utf-8')
    # with open('page_content.txt', 'w+', encoding='utf-8') as o:
    #     o.write(content)
    driver.get_screenshot_as_file(filepath)
    driver.quit()
    return filepath

def crop_image(filepath):
    crop_box = (46, 189, 661,804)
    new_size = (262,262)
    try:
        img = Image.open(filepath)
    except:
        print(f'Could not open file {filepath}')

    cropped = img.crop(crop_box)
    downsampled = cropped.resize(new_size, resample=Image.LANCZOS)
    img.close()
    cropped.close()
    downsampled.save(filepath)
    return filepath

if __name__ == '__main__':
    file = extract_screenshot()
    updated_file = crop_image(file)