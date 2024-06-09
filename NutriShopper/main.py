import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Set up model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Load the pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

# Explicitly set device to CPU
cfg.MODEL.DEVICE = "cpu"

# Create predictor
predictor = DefaultPredictor(cfg)



from bs4 import BeautifulSoup
from flask import Flask, render_template, jsonify, request
import requests
import json
from PIL import Image
import pytesseract
import pyfirmata
import time
import cv2 
import os
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'



app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


class TextExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
    
    def open_image(self):
        return Image.open(self.image_path)
    
    def extract_text(self, image):
        return pytesseract.image_to_string(image)
    
    def clean_text(self, text):
        return text.strip()
    
    def print_text(self, text):
        print("Extracted Text:")
        print(text)


def extract_text_from_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Perform OCR on the image
        text = pytesseract.image_to_string(img)
        return text.strip()

def search_google(query):
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = []

    # Extract search result titles and URLs
    for result in soup.find_all('div', class_='tF2Cxc'):
        title = result.find('h3').get_text()
        url = result.find('a')['href']
        search_results.append({'title': title, 'url': url})

    return search_results
@app.route('/llmoutput',methods=['POST'])
def llm(food):
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYTA5NDU3NjYtZThhMC00YWViLThjY2ItNjQxMGQ5YjJkOTNkIiwidHlwZSI6ImFwaV90b2tlbiJ9.6bBDFaJt31ekb8qG_sMC0OybxmN2b_iTM_z_-L16DZM"}

    url = "https://api.edenai.run/v2/text/chat"
    payload = {
        "providers": "openai",
        "text": "To make a balanced healthy diet, what are some foods that are healthier but similar to"+food ,
        "chatbot_global_action": "act as a food reccomender",
        "previous_history": [],
        "temperature": 0.0,
        "max_tokens": 20,
        "fallback_providers": ""
    }
    
    response = requests.post(url, json=payload, headers=headers)

    result = json.loads(response.text)
    rec = result['openai']['generated_text']
    print(rec)
    return "hiiii"



# Route to render the HTML page with the user input box
@app.route('/') 
def index():
    return render_template('index.html')

# Route to handle autocomplete requests
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query')

    # Make a request to the NutritionX API to get autocomplete suggestions
    api_url = 'https://trackapi.nutritionix.com/v2/search/instant'
    headers = {
        'x-app-id': '2fd322a3',
        'x-app-key': '4b8e69087f3dcbbac39bb2d2ab5eafc1'

    }
    params = {
        'query': query
    }

    response = requests.get(api_url, params=params, headers=headers)
    data = response.json()

    # Extract autocomplete suggestions from the API response
    suggestions = []
    if 'common' in data:
        for item in data['common']:
            suggestions.append(item['food_name'])

    return jsonify(suggestions)


def capture_image():
    try:
        camera = cv2.VideoCapture(0)
        return_value, image = camera.read()
        if not return_value:
            image = camera.read()
        
        # Check image dimensions
        if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
            cv2.imwrite('temp_image.jpg', image)
            del(camera)
            return 'temp_image.jpg'
        else:
            image = camera.read()
    except Exception as e:
        print("Error capturing image:", e)
        return None


@app.route('/capture', methods=['POST'])
def capture():
    image_path = capture_image()  # Capture image from webcam
    # text = extract_text_from_image(image_path)  # Extract text from the captured image
    #search_results = search_google(text)
    
    # Predict using the DefaultPredictor
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(image_path)  # Read the captured image
    outputs = predictor(im)  # Perform prediction

    os.remove(image_path)  # Delete the captured image

    # Convert outputs to a JSON-serializable format
    output_dict = outputs["instances"].to("cpu").get_fields()
    results = {
        "scores": output_dict["scores"].tolist(),
        "classes": output_dict["pred_classes"].tolist(),
        "boxes": output_dict["pred_boxes"].tensor.tolist()
    }

    return jsonify({'detection_results': results})






# Route for searching selected food
@app.route('/search', methods=['GET'])
def search_food():
    selected_food = request.args.get('food')

    # Make a request to the NutritionX API to get information about the selected food
    api_url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
    headers = {
        'x-app-id': '2fd322a3',
        'x-app-key': '4b8e69087f3dcbbac39bb2d2ab5eafc1',

        'Content-Type': 'application/json'
    }
    data = {
        'query': selected_food
    }

    response = requests.post(api_url, json=data, headers=headers)
    data = response.json()

    # Extract relevant information from the API response
    food_info = {
        'name': selected_food,
        'calories': data['foods'][0]['nf_calories'],
        'protein': data['foods'][0]['nf_protein'],
        'carbs': data['foods'][0]['nf_total_carbohydrate'],
        'fat': data['foods'][0]['nf_total_fat']
    }

    return jsonify(food_info)

if __name__ == '__main__':
    app.run(debug=True)

