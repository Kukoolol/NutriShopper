import torch
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Load the trained model
model = init_detector(config='/Users/naren/GroceriesX/Proj', checkpoint='/Users/naren/GroceriesX/Food Recognition Model.pth')

# Load and preprocess the input image
img = 'path_to_input_image.jpg'

# Perform inference
result = inference_detector(model, img)

# Visualize the result
show_result_pyplot(model, img, result)

