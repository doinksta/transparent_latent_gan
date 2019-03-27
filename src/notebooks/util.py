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
    
import pika
import json

""" make sure this notebook is running from root directory """
while os.path.basename(os.getcwd()) in ('notebooks', 'src'):
    os.chdir('..')
assert ('README.md' in os.listdir('./')), 'Can not find project root, please cd to project root before running the following code'

import src.tl_gan.generate_image as generate_image
import src.tl_gan.feature_axis as feature_axis
import src.tl_gan.feature_celeba_organize as feature_celeba_organize

from src.notebooks.conversions import feature_to_image, \
    b64_image_to_feature, b64_image_to_prediction_image, \
    dict_to_image, convert_naming_backend_to_frontend


if __name__ == "__main__":
    path_model = './asset_model/karras2019stylegan-ffhq-1024x1024.pkl'

    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))

    channel = connection.channel()

    channel.queue_declare(queue='rpc_queue')

    def on_request(ch, method, props, body):
        # print(body)
        n = json.loads(body)
        response = None
        print(type(body))
        print(" [.] fib(" + str(body)[:500] + ")")

        if n["route"] == "seed-image-recon":
            """
            Find the attributes and latent vector of the input image,
            and generate that face using the GAN.
            """
            data = n["data"]
            if type(data) == dict:
              print("lol wut data is a dict", data.keys())
              response = {}
            else:
              im = b64_image_to_prediction_image(data)
              response = {"data": im.decode('ascii')}

        elif n["route"] == "seed-image-features":
            """
            Find the attributes of the input image, and return it
            """
            data = n["data"]
            feature_dict = convert_naming_backend_to_frontend(
              b64_image_to_feature(data))

            # Recreate the dictionary form
            response = feature_dict

        elif n["route"] == "generate-image":
            """
            Generate an image given the input attributes
            """
            print(body)
            feature_dict = n["features"]
            im = dict_to_image(feature_dict)
            response = {"data": im.decode('ascii')}
        else:
            raise Exception("not a real route")

        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id = \
                                                             props.correlation_id),
                         body=json.dumps(response))
        ch.basic_ack(delivery_tag = method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(on_request, queue='rpc_queue')

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()

    """
    for i in range(1):
        for j in range(20):
            feature_to_image([0] * 19 + [(10 - j) * 0.005] + [0] * (20), save_name='src/notebooks/out/test_{}th_val_{}.jpg'.format(i, (5 - j) * 0.05))
    """
    """

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
        global Gs
        print('in func', Gs)
        im = dict_to_image(content, Gs)
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
    """
    '''
    from flask import Flask, request
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
        im = feature_to_image(content)

        return jsonify({"data": im})


    @app.route('/im_to_im', methods=['GET'])
    def im_to_im():
        content = request.json
        ret = b64_image_to_prediction_image(content['data'])
        return ret


    @app.route('/nearest_ims', methods=['GET'])
    def nearest_ims():
        content = request.json
        target_features = b64_image_to_feature(content['data'])

        pd.read_csv('data/processed/UTKFace_features.csv')

        def euclidean_distance():
            pass
        return ret
    '''
