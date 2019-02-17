import os
import glob
import sys
import numpy as np
import pickle
import tensorflow as tf
import PIL
import ipywidgets
import io
import pandas as pd

""" make sure this notebook is running from root directory """
while os.path.basename(os.getcwd()) in ('notebooks', 'src'):
    os.chdir('..')
assert ('README.md' in os.listdir('./')), 'Can not find project root, please cd to project root before running the following code'

import src.tl_gan.generate_image as generate_image
import src.tl_gan.feature_axis as feature_axis
import src.tl_gan.feature_celeba_organize as feature_celeba_organize
from src.notebooks.ranker import get_top_ten

from src.notebooks.conversions import feature_to_image as feature_to_image, b64_image_to_feature, b64_image_to_prediction_image, dict_to_image

from flask import Flask, request, jsonify
app = Flask(__name__)


@app.route('/im_to_feat', methods=['GET'])
def im_to_feat():
    content = request.json
    raw_features = b64_image_to_feature(content['data'])
    
    # Recreate the dictionary form    
    ret = {}
    
    for label, value in zip(feature_celeba_organize.feature_name_celeba_org, raw_features):
        ret[label] = value
    
    return jsonify(ret)


@app.route('/feat_to_im', methods=['GET'])
def feat_to_im():
    content = request.json
    print(content)
    im = dict_to_image(content)
    return jsonify({"data": im})


@app.route('/im_to_im', methods=['GET'])
def im_to_im():
    content = request.json
    im = b64_image_to_prediction_image(content['data'])
    return jsonify({"data": im})


@app.route('/nearest_ims', methods=['GET'])
def nearest_ims():
    content = request.json
    target_features = b64_image_to_feature(content['data'])
    
    return jsonify({"data": get_top_ten(target_features)})

app.run()