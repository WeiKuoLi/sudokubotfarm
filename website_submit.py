from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time
import csv
import numpy as np

#Start by importing solution matrix from previous program
puzzsolve = np.loadtxt('solution.csv',delimiter=',',dtype=int)

#Now open webpage and input answers

def input_solution(web_url:str):
    options = Options()
    options.headless = True
    
    options.add_argument("--window-size=1280,820")
    print("STUCK HERE")
    driver = webdriver.Chrome(options=options)
    print("STUCK HERE")
    driver.get(web_url)

    actions = ActionChains(driver)
    actions.move_by_offset(300,320).click().perform() #moves to 320,300 pixel and clicks mouse
    time.sleep(5)

    driver.quit()
    print("Can you see me???")
    return True

input_solution('https://sudoku.com/expert/')