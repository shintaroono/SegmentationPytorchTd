### test to print hello world. ###
# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def hello():
#     return 'Hello World!'


### just reference. ###
# @app.route('/predict', methods=['POST'])
# def predict():
    # return 'Hello World!'




### send json data test. url in td should be like 'http://localhost:5000/predict'. ###
# from flask import Flask, jsonify
# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})




### inference test. ###
# import io

# import torchvision.transforms as transforms
# from PIL import Image

# # method takes image data in bytes, applies the series of transforms and returns a tensor. #
# def transform_image(image_bytes):
#     my_transforms = transforms.Compose([transforms.Resize(255),
#                                         transforms.CenterCrop(224),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(
#                                             [0.485, 0.456, 0.406],
#                                             [0.229, 0.224, 0.225])
#                                         ])
#     image = Image.open(io.BytesIO(image_bytes))
#     return my_transforms(image).unsqueeze(0)

# # open sample data as tensor using transform_image() method. #
# # with open("tutorials\_static\img\sample_file.jpeg", 'rb') as f:
# #     image_bytes = f.read()
# #     tensor = transform_image(image_bytes=image_bytes)
# #     print(tensor)


# # prediction. #
# from torchvision import models

# model = models.densenet121(pretrained=True)
# model.eval()

# import json

# imagenet_class_index = json.load(open('tutorials\_static\imagenet_class_index.json'))

# # y_hat will contain the index of the predicted class id. above json file contains the mapping of ImageNet class id to ImageNet class name. #
# def get_prediction(image_bytes):
#     tensor = transform_image(image_bytes=image_bytes)
#     outputs = model.forward(tensor)
#     _, y_hat = outputs.max(1)
#     predicted_idx = str(y_hat.item())
#     return imagenet_class_index[predicted_idx]

# # open test the above method. #
# with open("tutorials\_static\img\sample_file.jpeg", 'rb') as f:
#     image_bytes = f.read()
#     # The first item in array is ImageNet class id and second item is the human readable name. #
#     print(get_prediction(image_bytes=image_bytes))




### integrating the model in flask api server. ###
import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
imagenet_class_index = json.load(open('tutorials\_static\imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run()
