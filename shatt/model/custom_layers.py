'''
Created on 10.04.2019

@author: Philipp
'''
import tensorflow as tf
from tensorflow.keras.layers import Layer


class AlphaRegularization(Layer):
    
    def __init__(self, alpha_c, caption_max_length, image_features_dimensions, **kwargs):
        self.alpha_c = alpha_c
        self.caption_max_length = caption_max_length
        self.image_features_dimensions = image_features_dimensions
        self.kwargs = kwargs
        super(AlphaRegularization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.caption_max_length = tf.cast(self.caption_max_length, dtype="float32")
        self.image_features_dimensions = tf.cast(self.image_features_dimensions, dtype="float32")
        self.ratio = self.caption_max_length / self.image_features_dimensions
        super(AlphaRegularization, self).build(input_shape)

    def call(self, x):
        output = x[0]
        attentions = x[1]
        attentions = tf.transpose(attentions, (1, 0, 2))
        attentions_sum = tf.reduce_sum(attentions, axis=1) # sum over the batches
        attentions_sum = self.ratio - attentions_sum # caption_length and feature_dimension
        attentions_reg = self.alpha_c * tf.reduce_sum(attentions_sum ** 2)
        #attentions_reg = tf.Print(attentions_reg, [attentions_reg])
        self.add_loss(attentions_reg, attentions)
        return output
    
    def compute_output_shape(self, input_shape):
        shape = input_shape[0]
        return tf.TensorShape(shape)
    
    def get_config(self):
        base_config = super(AlphaRegularization, self).get_config()
        base_config['alpha_c'] = self.alpha_c
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ReduceMean(Layer):
    
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(ReduceMean, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReduceMean, self).build(input_shape)

    def call(self, x):
        return tf.reduce_mean(x, axis=self.axis)
    
    def compute_output_shape(self, input_shape):
        shape_dims = [s for s in range(len(input_shape)) if s != self.axis]
        shape = [input_shape[d] for d in shape_dims]
        return tf.TensorShape(shape)
    
    def get_config(self):
        base_config = super(ReduceMean, self).get_config()
        base_config['axis'] = self.axis
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class SamplingLayer(Layer):
    
    def __init__(self, use_argmax, **kwargs):
        self.use_argmax = use_argmax
        super(SamplingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SamplingLayer, self).build(input_shape)

    def call(self, x):
        if self.use_argmax:
            return tf.argmax(x, 1, name="argmax_sampling")
        return tf.random.categorical(tf.log(x), 1, name="categorical_sampling")
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, 1])
    
    def get_config(self):
        base_config = super(SamplingLayer, self).get_config()
        base_config["use_argmax"] = self.use_argmax
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


CUSTOM_LAYER_REGISTRY = {
            "ReduceMean": ReduceMean,
            "SamplingLayer": SamplingLayer
}
