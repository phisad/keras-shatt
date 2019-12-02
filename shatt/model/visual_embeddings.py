'''
Created on 05.03.2019
                      
@author: Philipp
'''
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Reshape, Input, Dense, Dropout

def image_features_model(output_layer, image_feature_size, input_shape, image_top_layer, image_top_layer_dropout_rate):
    image_features, vinput = image_features_graph_flatten(output_layer, image_feature_size, input_shape, image_top_layer, image_top_layer_dropout_rate)
    return Model(vinput, image_features)

    
def __add_top_layer(outputs, dropout_rate):
    print("Adding top layer on image feature extractor with dropout", str(dropout_rate))
    outputs = Dense(512, activation="tanh", name="image_features_top")(outputs)
    outputs = Dropout(rate=dropout_rate, name="image_features_dropout")(outputs)
    return outputs


def bypass_image_features_graph_flatten(image_feature_size, image_top_layer, image_top_layer_dropout_rate):
    vinput = Input((image_feature_size, 512), name="input_images")
    outputs = vinput
    if image_top_layer:
        outputs = __add_top_layer(outputs, image_top_layer_dropout_rate)
    return outputs, vinput
    

def image_features_graph_flatten(output_layer, image_feature_size, input_shape, image_top_layer, image_top_layer_dropout_rate):
    inputs, outputs = __image_features_graph(output_layer, input_shape)
    outputs = Reshape((image_feature_size, 512), name="flatten_filters")(outputs)
    if image_top_layer:
        outputs = __add_top_layer(outputs, image_top_layer_dropout_rate)
    return outputs, inputs


def __image_features_graph(output_layer, input_shape):
    vinput = Input(input_shape, name="input_images")
    base_model = VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=vinput,
        pooling=None,
        classes=None)
    for layer in base_model.layers:
        layer.trainable = False
    return base_model.input, base_model.get_layer(output_layer).output
