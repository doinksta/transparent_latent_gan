import os
import glob
import sys
import numpy as np
import pickle
import pandas as pd
import h5py
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.applications
import tensorflow.keras.layers as layers
from tensorflow.keras.applications.mobilenet import preprocess_input
import ipywidgets
from io import BytesIO
from PIL import Image
import base64


""" make sure this notebook is running from root directory """
while os.path.basename(os.getcwd()) in ('notebooks', 'src'):
    os.chdir('..')
assert ('README.md' in os.listdir('./')), 'Can not find project root, please cd to project root before running the following code'

import src.tl_gan.generate_image as generate_image
import src.tl_gan.feature_axis as feature_axis
import src.tl_gan.feature_celeba_organize as feature_celeba_organize

# Statics

# Create tf session
yn_CPU_only = False
if yn_CPU_only:
    config = tf.ConfigProto(device_count = {'GPU': 0}, allow_soft_placement=True)
else:
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)

# Load feature directions and labels
path_feature_direction = './src/asset_results/stylegan_feature_direction_40'

pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction']
feature_names = feature_direction_name['name']



# path to model code and weight
path_style_gan_code= './src/model/stylegan'
path_model = './src/asset_model/karras2019stylegan-ffhq-1024x1024.pkl'
sys.path.append(path_style_gan_code)
try:
    with open(path_model, 'rb') as file:
        G, D, Gs = pickle.load(file)
except FileNotFoundError:
    print('before running the code, download pre-trained model to project_root/asset_model/')
    raise


# more code
path_model_save = './src/asset_model/cnn_face_attr_celeba'


"""
Takes in a 40-D description of an individual, converts this into a 512-D space
(using 'feature_direction', and returns an image of the predicted indvidual.

@param  features: a list of 40 values
returns: an image as a result of the prediction model
"""
def feature_to_image(features, feature_direction=feature_direction, save_name=None):
    feature_lock_status = np.zeros(len(feature_direction)).astype('bool')
    print(features)
    print(feature_direction)
    print(type(feature_direction))
    
    feature_directoion_disentangled = feature_axis.disentangle_feature_axis_by_idx(
        feature_direction, idx_base=np.flatnonzero(feature_lock_status))
    
#    z_sample = np.random.randn(512)
    z_sample = np.array([0] * 512)
    feature_direction_transposed = np.transpose(feature_direction)

    step_size = 0.01
    
    for direction, feature_val, idx_feature in zip(feature_direction_transposed, features, range(len(features))):
        # print(z_sample, feature_val, feature_directoion_disentangled[:, idx_feature], step_size)
        z_sample = np.add(z_sample, feature_val * feature_directoion_disentangled[:, idx_feature] * step_size)
    
    sess = tf.InteractiveSession(config=config)
    x_sample = generate_image.gen_single_img(z=z_sample, Gs=Gs)
    
    if save_name != None:
        generate_image.save_img(x_sample, save_name)
    

    im = Image.fromarray(x_sample)
    

    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
        
    return img_str

    
"""
Helper method to take in a feature dictionary that is partially filled, and generate a prediction image from it.
"""
def dict_to_image(feature_dict, feature_names=feature_names):
    features = []
    
    for feature_name in feature_names:
        if feature_name in feature_dict.keys():
            features.append(feature_dict.get(feature_name))
        else:
            features.append(0)
    return feature_to_image(features)

    
def create_cnn_model(size_output=None, tf_print=False):
    """
    create keras model with convolution layers of MobileNet and added fully connected layers on to top
    :param size_output: number of nodes in the output layer
    :param tf_print:    True/False to print
    :return: keras model object
    """

    if size_output is None:
        # MAKE SURE TO PASS IN HARDCODED SIZE - DON'T HAVE CSV DATA FILES
        # get number of attrubutes, needed for defining the final layer size of network
        df_attr = pd.read_csv(path_celeba_att, sep='\s+', header=1, index_col=0)
        size_output = df_attr.shape[1]

    # Load the convolutional layers of pretrained model: mobilenet
    base_model = tf.keras.applications.mobilenet.MobileNet(include_top=False, input_shape=(128,128,3),
                                                          alpha=1, depth_multiplier=1,
                                                          dropout=0.001, weights="imagenet",
                                                          input_tensor=None, pooling=None)

    # add fully connected layers
    fc0 = base_model.output
    fc0_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='fc0_pool')(fc0)
    fc1 = layers.Dense(256, activation='relu', name='fc1_dense')(fc0_pool)
    fc2 = layers.Dense(size_output, activation='tanh', name='fc2_dense')(fc1)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=fc2)

    # freeze the early layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='sgd', loss='mean_squared_error')

    if tf_print:
        print('use convolution layers of MobileNet, add fully connected layers')
        print(model.summary())

    return model

def get_list_model_save(path_model_save=path_model_save):
    return glob.glob(os.path.join(path_model_save, 'model*.h5'))

model = create_cnn_model(size_output=40)
model.load_weights(get_list_model_save()[-1])

def image_to_feature(image_path, model=model):
    img = np.asarray(PIL.Image.open(image_path).resize((128, 128), resample=PIL.Image.BILINEAR))
    x = np.stack([img], axis=0)
    return model.predict(preprocess_input(x))

def b64_image_to_feature(b64_str, model=model):
    image_data = re.sub('^data:image/.+;base64,', '', b64_str)
    img = np.asarray(PIL.Image.open(BytesIO(base64.b64decode(image_data))).resize((128, 128), resample=PIL.Image.BILINEAR))
    x = np.stack([img], axis=0)
    raw_features = model.predict(preprocess_input(x)).flatten()
    
    # Recreate the dictionary form    
    ret = {}
    
    for label, value in zip(feature_celeba_organize.feature_name_celeba_org, raw_features):
        ret[label] = value
    
    return ret


def b64_image_to_prediction_image(b64_str):
    return feature_to_image(b64_image_to_feature(b64_str))

"""if __name__ == "__main__":
    np.array(image_to_feature("data/processed/cyberextender/000420/000007.jpg")[0])

    feature_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    feature_list= []
    for i, img_name in enumerate(glob.glob("./data/processed/UTKFace/*.jpg")):
        feature_list.append(image_to_feature(img_name)[0])
        if i % 100 == 0:
            print(i)

    df = pd.DataFrame(feature_list, columns=feature_names)
    df.to_csv("./data/processed/UTKFace_features.csv")

    print(df.head())"""