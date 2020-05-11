### test to print hello world. ###
# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def hello():
#     return 'Hello World!'


# ### just reference. ###
# @app.route('/predict', methods=['POST'])
# def predict():
    # return 'Hello World!'




### send json data test. url in td should be like 'http://localhost:5000/predict'. ###
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})
