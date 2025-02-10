import requests
import cv2
import numpy as np 
import pytesseract
from PIL import Image
import pytesseract
import os


def preprocess_image_for_ocr(imagePath):
    image = cv2.imread(imagePath)
   # cv2.imshow('Image',image)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
   # cv2.waitKey(0)
    # closing all open windows
   # cv2.destroyAllWindows()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converts to grayscale
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  # Resize image
    gray = cv2.medianBlur(gray, 9)  # Reduces noise
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 6)  # Enhance text
   # cv2.imshow('Image',gray)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
   # cv2.waitKey(0)
    # closing all open windows
    #cv2.destroyAllWindows()
    return gray



def json(metin):
    url = 'http://localhost:11434/api/generate'
    headers = {'Content-Type': 'application/json'}    
    data = {"model": "llama3.2","prompt":"Given OCR data from a receipt,I need you to structure it into a well-organized JSON format."+metin,"stream":False}
    response = requests.post(url, headers=headers, json=data)
    return print(response.json()["response"])

def extract_text_from_image(image):
    processed_image = preprocess_image_for_ocr(image)
    text = pytesseract.image_to_string(processed_image, config='--oem 1 --psm 4')
    a= json(text)
    return a


extract_text_from_image("j.jpeg")
