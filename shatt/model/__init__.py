from shatt.model.visual_embeddings import bypass_image_features_graph_flatten, \
    image_features_graph_flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, RNN, LSTMCell, Embedding, Flatten, Lambda, Reshape, Concatenate
from tensorflow.keras.layers import Add, Activation, Dot, TimeDistributed, Dropout, Multiply
from shatt.model.custom_layers import ReduceMean, SamplingLayer, \
    AlphaRegularization
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.layers.normalization import BatchNormalization


def create_shatt_model_from_config(config, inference_mode, return_inference_model=False, 
                                   attention_graph=None, dynamic_attention=False, 
                                   use_input_attention=False, attention_iterations=None):
    print("Dropout rate: {:.2f}".format(config.getDropoutRate()))
    print("Alpha regularizer: {:.4f}".format(config.getAlphaRegularization()))
    image_features_size = config.getImageFeaturesSize()
    print("Image Feature size: {:}".format(image_features_size))
    use_max_sampler = config.getUseMaxSampler()
    print("Word sampler is: {}".format("argmax" if use_max_sampler else "prob"))
    if config.getByPassImageFeaturesComputation():
        image_features_graph = bypass_image_features_graph_flatten(image_features_size, config.getImageTopLayer(), config.getImageTopLayerDropoutRate())
        print("Bypass image feature generation")
    else:
        output_layer = config.getImageOutputLayer()
        image_features_graph = image_features_graph_flatten(output_layer, image_features_size, config.getImageInputShape(), config.getImageTopLayer(), config.getImageTopLayerDropoutRate())

    return create_shatt_model_v2(image_features_graph, config.getCaptionMaximalLength(), config.getVocabularySize(),
                                 config.getDropoutRate(), config.getVocabularyStartSymbol(),
                                 config.getImageFeaturesSize(), config.getAlphaRegularization(),
                                 attention_graph=attention_graph, # (attention_output, attention_input)
                                 dynamic_attention=dynamic_attention, # if to combine alternating attention and attention graph
                                 use_input_attention=use_input_attention, # if to use the attention_input
                                 attention_iterations=attention_iterations, # how many iterations to use the attention_input
                                 inference_mode=inference_mode, return_inference_model=return_inference_model,
                                 use_max_sampler=use_max_sampler)


def create_shatt_model_v2(image_features_graph, caption_max_length, vocabulary_size, dropout_rate,
                              start_encoding, image_features_dimensions, alpha_reg=.0,
                              embedding_size=512, hidden_size=1024, inference_mode=False,
                              attention_graph=None, return_attention=False, return_inference_model=False,
                              use_max_sampler=True, dynamic_attention=False, 
                              use_input_attention=False, attention_iterations=None):
    builder = ModelBuilder(caption_max_length, vocabulary_size, dropout_rate, start_encoding, image_features_dimensions, embedding_size, hidden_size, alpha_reg, use_max_sampler)
    return builder.create_shatt_model(image_features_graph, attention_graph, return_attention, inference_mode, return_inference_model, dynamic_attention, use_input_attention, attention_iterations)

    
class ModelBuilder():
    
    def __init__(self, caption_max_length, vocabulary_size,
                 dropout_rate, start_encoding, image_features_dimensions,
                 embedding_size=512, hidden_size=1024, alpha_reg=.005, use_max_sampler=True):
            """ arguments """
            self.caption_max_length = caption_max_length
            self.vocabulary_size = vocabulary_size
            self.dropout_rate = dropout_rate
            self.embedding_size = embedding_size
            self.hidden_size = hidden_size
            self.image_features_dimensions = image_features_dimensions
            
            """ the input captions for teacher forcing during training (these are ignored in inference mode)"""
            self.input_captions = Input(name="input_captions", shape=(caption_max_length, 1))
            self.input_image_normalize = BatchNormalization(momentum=.95, name="shatt_image_batch_normalize")
            
            """ initial image projection layer """
            self.image_projector_shape = Reshape(target_shape=(embedding_size,))
            self.image_projector = Dense(embedding_size, use_bias=False)
            
            """ attention layer (attend) """
            self.state_projector = Dense(embedding_size, activation=None, name="shatt_image_projector")
            self.state_projector_add = Add(name="shatt_image_projection_sum")
            self.state_projector_activation = Activation("relu", name="shatt_image_projection")
            
            spatial_kernel = Dense(1, activation=None, name="shatt_image_attention_kernel")
            self.spatial_reductor = TimeDistributed(spatial_kernel, name="shatt_image_reduction")
            self.spatial_flatten = Flatten(name="shatt_image_attention_flatten")
            self.spatial_attention = Activation("softmax", name="shatt_image_attention")
            self.spatial_attention_feature = Dot(axes=(1, 1), name="shatt_image_attention_feature")
            
            self.attention_regularizer = AlphaRegularization(alpha_reg, caption_max_length, image_features_dimensions,
                                                             name="shatt_image_attention_regularizer")
            
            """ select layer (beta) """
            self.image_context_attention = Dense(1, activation="sigmoid", name="shatt_image_context_attention")
            self.image_context_attention_feature = Multiply(name="shatt_image_context_attention_feature")
    
            """ decode layer (tell) """
            self.decode_state_dropout = Dropout(rate=dropout_rate, name="shatt_decode_state_dropout")
            self.decode_state_predictor = Dense(embedding_size, activation=None, name="shatt_decode_state_predictor")
            self.decode_attention_predictor = Dense(embedding_size, activation=None, use_bias=False, name="shatt_decode_attention_predictor")
            self.decode_combiner = Add(name="shatt_decode_caption_embedding")
            self.decode_caption_predictor = Dense(vocabulary_size + 1, "softmax", name="shatt_decode_caption_predictor")
            self.decode_caption_sampler = SamplingLayer(use_argmax=use_max_sampler, name="shatt_decode_caption_sampling")
            
            """ embedding layer """
            self.embedding = Embedding(input_dim=vocabulary_size + 1,  # b.c. of padding value 0
                                       output_dim=embedding_size,
                                       mask_zero=False,
                                       name="shatt_word_embeddings")
            self.embedding_flatten_layer = Flatten(name="shatt_cell_embedding_flatten")
            
            """ recurrent layer """
            self.lstm = LSTMCell(hidden_size, name="shatt_internal_lstm")
            self.lstm_input_layer = Concatenate(name="shatt_cell_lstm_inputs")
            
            """ zero like layer to dynamically initialize the previous caption in the first time step based on the batch size"""
            self.zeros_layer = Lambda(lambda x: K.ones_like(x, dtype="float32") * start_encoding, name="shatt_cell_caption_initial")
            
            """ reshaping layer for the output, so that they can be concatenated easily """
            self.output_reshape_caption_layer = Reshape(target_shape=(1, 1), name="shatt_cell_outputs_caption_reshape")
            self.output_reshape_probs_layer = Reshape(target_shape=(1, vocabulary_size + 1), name="shatt_cell_outputs_probs_reshape")
            self.output_concatenate_layer = Concatenate(axis=1, name="shatt_cell_outputs_concatenate")
            
            """ reshaping layer for the output, so that they can be concatenated easily """
            self.output_attention_reshape_layer = Reshape(target_shape=(1, -1), name="shatt_cell_outputs_attention_reshape")
            self.output_attention_concatenate_layer = Concatenate(axis=1, name="shatt_cell_outputs_attention_concatenate")
            
            """ initalizer layers """
            self.input_features_reductor = ReduceMean(axis=1, name="average_image_features")
            self.generator_state_initializer = Dense(hidden_size, "tanh", name="initial_generator_state")
            self.generator_context_initializer = Dense(hidden_size, "tanh", name="initial_generator_context") 
    
    def __build_model(self, image_features_graph, attention_graph=None,
                      return_attention=False, inference_mode=False,
                      dynamic_attention=False, use_input_attention=False, 
                      attention_iterations=None):    
        """ the other features graphs have to return both, the outputs and the inputs """
        if use_input_attention and attention_graph == None:
            input_attention = Input(shape=(self.image_features_dimensions,), name="input_attention")
            # set attention graph here, when input_attention is requested
            # that that is set as attention_features next
            attention_graph = input_attention, input_attention
            
        if attention_graph == None:
            image_features, input_images = image_features_graph
        else:
            image_features, input_images = image_features_graph
            attention_features, input_attention = attention_graph
        
        image_features = self.input_image_normalize(image_features)
            
        with K.name_scope("shatt_generator_initial"):
            with K.name_scope("shatt_generator_initial_lstm"):
                igraph = self.input_features_reductor(image_features)
                initial_state = self.generator_state_initializer(igraph)
                initial_context = self.generator_context_initializer(igraph)
                
                previous_states = [initial_state, initial_context]
                previous_state = initial_state
                previous_caption = None
            
            with K.name_scope("shatt_generator_initial_projection"):
                initial_features_dimensions = image_features.shape[1]
                image_features_projected = self.image_projector_shape(image_features)
                image_features_projected = self.image_projector(image_features_projected)
                image_features_projected = Reshape(target_shape=(initial_features_dimensions, self.embedding_size))(image_features)
                
        for timestep in range(self.caption_max_length):  # one more timestep for end tag
            with K.name_scope("shatt_cell_step"):
                """
                    The argument 'constants' must be added to support constants in RNN.
                """
                with K.name_scope("shatt_cell_input_slice"):
                    step_suffix = "_{}".format(str(timestep))
                    # externalize this to single layer somehow leads to different lstm state shapes
                    # maybe because timestep is not coming from a layer (we should make a custom layer that takes timestep as state)
                    caption = Lambda(lambda x: x[:, timestep, :], name="shatt_input_caption_slice" + step_suffix)(self.input_captions)
                
                with K.name_scope("shatt_cell_attend"):
                    if attention_graph == None or dynamic_attention or (attention_iterations and timestep >= attention_iterations):
                        # TODO the state projection should not go in each timestep, but only before the first?
                        current_state_projection = self.state_projector(previous_state)
                        current_state_projection = self.state_projector_add([image_features_projected, current_state_projection])
                        current_state_projection = self.state_projector_activation(current_state_projection)
                        
                        current_image_attention = self.spatial_reductor(current_state_projection)
                        current_image_attention = self.spatial_flatten(current_image_attention)
                        current_image_attention = self.spatial_attention(current_image_attention)
                        
                        if dynamic_attention:
                            print("Use dynamic attention at timestep", timestep)
                            current_image_attention = attention_features([current_image_attention, input_attention])
                        else:
                            print("Use alternating attention at timestep", timestep)
                    else:
                        # use input attention
                        print("Use attention features at timestep", timestep)
                        current_image_attention = attention_features
                    
                    """ we collect the attention vectors for both regularization and introspection """    
                    model_attention_output = self.output_attention_reshape_layer(current_image_attention)
                    if timestep == 0:
                        model_attention_outputs = model_attention_output
                    else:
                        model_attention_outputs = self.output_attention_concatenate_layer([model_attention_outputs, model_attention_output])
                    
                    current_image_context = self.spatial_attention_feature([image_features, current_image_attention])
                
                with K.name_scope("shatt_cell_adjust"):
                    """ 
                        weight the image context vector based on the previous state
                        this allows the model to adjust the importance of the image
                        based on the previous state (predicting beta gating scalar)
                    """
                    current_image_context_selector = self.image_context_attention(previous_state)
                    current_selected_image_context = self.image_context_attention_feature([current_image_context_selector, current_image_context])
                    
                with K.name_scope("shatt_cell_encode"):    
                    if previous_caption == None:
                        with K.name_scope("shatt_cell_initial"):
                            initial_previous_caption = self.zeros_layer(caption)
                            embedded_previous_caption = self.embedding(initial_previous_caption)
                            embedded_previous_caption = self.embedding_flatten_layer(embedded_previous_caption)
                    else:
                        with K.name_scope("shatt_cell_step"):
                            embedded_previous_caption = self.embedding(previous_caption)
                            embedded_previous_caption = self.embedding_flatten_layer(embedded_previous_caption)
            
                with K.name_scope("shatt_cell_lstm"):
                    """ Notice: this might throw an exception, when the last batch is smaller than the previous one """
                    current_inputs = self.lstm_input_layer([embedded_previous_caption, current_selected_image_context])
                    current_state, *states = self.lstm(inputs=current_inputs, states=previous_states)
                
                with K.name_scope("shatt_cell_decode"):
                    """ use current state """
                    decode_state = self.decode_state_dropout(current_state)
                    decode_state = self.decode_state_predictor(decode_state)
                    
                    """ add current image context """
                    decode_attention = self.decode_attention_predictor(current_selected_image_context)
                    decode_state = self.decode_combiner([decode_state, decode_attention])
                    
                    """ add previous caption embedding """
                    decode_state = self.decode_combiner([decode_state, embedded_previous_caption])
                    
                    """ predict output caption """
                    decode_state = self.decode_state_dropout(decode_state)
                    output_caption_probs = self.decode_caption_predictor(decode_state) 
                    output_captions = self.decode_caption_sampler(output_caption_probs)
                
                with K.name_scope("shatt_cell_link"):
                    if inference_mode:
                        previous_caption = output_captions
                    else:  # teacher forcing
                        previous_caption = caption
                    previous_states = states 
                    previous_state = current_state
                    
                with K.name_scope("shatt_cell_outputs"):
                    if inference_mode:
                        target_output = output_captions
                        target_output = self.output_reshape_caption_layer(target_output)
                    else:
                        target_output = output_caption_probs
                        target_output = self.output_reshape_probs_layer(target_output)
                        
                    if timestep == 0:
                        model_target_outputs = target_output
                    else:
                        """ Notice: this might throw an exception, when the last batch is smaller than the previous one """
                        model_target_outputs = self.output_concatenate_layer([model_target_outputs, target_output])
                    
                previous_states = states
                previous_state = current_state

        if attention_graph == None:
            model_inputs = [self.input_captions, input_images]
        else:
            model_inputs = [self.input_captions, input_images, input_attention]
        
        """ attach the regularizer to the model graph avoiding an extra model output """
        if not inference_mode:
            model_target_outputs = self.attention_regularizer([model_target_outputs, model_attention_outputs])
        
        if not return_attention: 
            model_outputs = model_target_outputs
        else:  # usually only in inference mode (this wont work automatically when training)
            model_outputs = [model_target_outputs, model_attention_outputs]
            
        return Model(inputs=model_inputs, outputs=model_outputs)

    def create_shatt_model(self, image_features_graph,
                           attention_graph=None, return_attention=False,
                           inference_mode=False, return_inference_model=False,
                           dynamic_attention=False, use_input_attention=False, attention_iterations=None):
        model = self.__build_model(image_features_graph, attention_graph, return_attention, inference_mode, dynamic_attention, use_input_attention, attention_iterations)
        if return_inference_model and not inference_mode:
            inference_model = self.__build_model(image_features_graph, attention_graph, return_attention,
                                                 return_inference_model, dynamic_attention, use_input_attention, attention_iterations)
            return [model, inference_model]
        return model
                
