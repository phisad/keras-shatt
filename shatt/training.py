'''
Created on 01.03.2019

The supervised training pipeline is as follows:

Preparation:
- pre-process input data
- determine a data provider
- determine a network architecture

Training:
- initialize the network variables as a trainable model
- forward propagate data through the model
- calculate loss based on the actual data labels
- back propagate the loss and update the network variables

Stopping:
- monitor the training progress by calculating metrics e.g. accuracy on a validation set 
- determine stopping strategy e.g. all data run through once or metrics not changing anymore
- stop and save the model e.g. each iteration or at the end

@author: Philipp
'''
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy

from shatt import SPLIT_TRAIN, SPLIT_VALIDATE
from shatt.scripts import OPTION_FORCE_SAVE
from shatt.sequences import MsCocoCaptionSequence, MscocoPredictionSequence
from shatt.dataset.vocabulary import PaddingVocabulary, store_vocabulary
from tensorflow.python.keras.optimizers import Adam
from shatt.model import create_shatt_model_from_config
from shatt.dataset.captions import load_prepared_captions_json_from_config, \
    to_caption_listing_by_image_id
from shatt.configuration import store_configuration
from shatt.callbacks import create_tensorboard_from_save_path, create_csv_logger, \
    create_checkpointer, create_scores_logger, predict, \
    EmbeddingProjectorCallback, TrainingScoresMerger


def print_model_summary(model, config):
    if config.getPrintModelSummary():
        print(model.summary())


def __get_sequences(config, split_name):
    if split_name:
        vocabulary = PaddingVocabulary.create_vocabulary_from_config(config, split_name)
        training_sequence = MsCocoCaptionSequence.create_mscoco_labelled_sequence_from_config(config, vocabulary, split_name)
        validation_sequence = None
    else:
        vocabulary = PaddingVocabulary.create_vocabulary_from_config(config, SPLIT_TRAIN)
        training_sequence = MsCocoCaptionSequence.create_mscoco_labelled_sequence_from_config(config, vocabulary, SPLIT_TRAIN)
        validation_sequence = MsCocoCaptionSequence.create_mscoco_labelled_sequence_from_config(config, vocabulary, SPLIT_VALIDATE)
    return training_sequence, validation_sequence


def __get_log_path(config, split_name):
    model_name = config.getModelType()
    is_top = "top" if config.getImageTopLayer() else "notop"
    split = split_name if split_name else "train"
    log_path = "{}/{}/{:.2f}/{}/{}".format(model_name, split, config.getDropoutRate(), is_top, config.getModelDerivateName())
    return log_path


def __get_callbacks(config, split_name):
    tensorboard_logdir = config.getTensorboardLoggingDirectory()
    model_log_path = __get_log_path(config, split_name)
    # update_frequency = "epoch"
    # if config.is_dryrun():
    update_frequency = "batch"
    tagged_log_path, tensorboard_logger = create_tensorboard_from_save_path(tensorboard_logdir, model_log_path, update_frequency)
    
    callbacks_listing = []
    # early stopping should actually be based on BLEU score
    callbacks_listing.append(tensorboard_logger)
    callbacks_listing.append(create_csv_logger(tagged_log_path))
    
    if not config.is_dryrun() or config[OPTION_FORCE_SAVE]:
        # checkpoint monitor should actually be based on the BLEU score (but this is not a measurement yet)
        checkpoint_saver = create_checkpointer(tagged_log_path, config.getModelType(), store_per_epoch=True)
        callbacks_listing.append(checkpoint_saver)
    
    return tagged_log_path, callbacks_listing


def __getOptimizer(config):
    optimizer = Adam()
    print("Apply {} optimizer".format(type(optimizer).__name__))
    return optimizer


def masked_categorical_crossentropy(y_true, y_pred):
    print("Apply masked_categorical_crossentropy")
    loss = categorical_crossentropy(y_true, y_pred)
    y_sparse_true = tf.argmax(y_true, axis=2, name="masked_loss_ytrue_sparse")
    mask = tf.cast(tf.not_equal(y_sparse_true, 0), dtype="float32")
    return loss * mask


def masked_categorical_accuracy(y_true, y_pred):
    print("Apply masked_categorical_accuracy")
    y_sparse_true = tf.argmax(y_true, axis=2, name="masked_accuracy_ytrue_sparse")
    y_sparse_pred = tf.argmax(y_pred, axis=2, name="masked_accuracy_ypred_sparse")
    
    y_correct = tf.cast(tf.equal(y_sparse_true, y_sparse_pred), dtype="float32")
    y_mask = tf.cast(tf.not_equal(y_sparse_true, 0), dtype="float32")
    
    y_correct_masked = tf.reduce_sum(y_correct * y_mask, axis=1)
    y_word_count = tf.reduce_sum(y_mask, axis=1)
    
    y_batch_size = tf.cast(tf.shape(y_correct_masked)[0], dtype="float32")
    y_masked_accuracy = tf.reduce_sum(y_correct_masked / y_word_count) / y_batch_size
    return y_masked_accuracy


def __get_model(config, path_to_model, initial_epoch, inference_mode=False, return_inference_model=False):
    """
        Loads the model from the given path or creates a new model based on the configuration.
        The model is compiled before return.
    """
    if path_to_model:
        if not config.is_dryrun() and not initial_epoch:
            raise Exception("You have to set the initial_epoch, when continuing training")
        
    print("Create shatt models")
    models = create_shatt_model_from_config(config, inference_mode, return_inference_model)
    
    if isinstance(models, list):
        print("Created training and inference model")
        training_model = models[0]
    else:
        training_model = models
        
    optimizer = __getOptimizer(config)
    
    if path_to_model:
        print("Try to load model weights from path: " + path_to_model)
        training_model.load_weights(path_to_model, by_name=True)
    else:
        print("Compile training model")
        training_model.compile(optimizer=optimizer, loss=masked_categorical_crossentropy, metrics=[masked_categorical_accuracy])
        
    print_model_summary(training_model, config)
    # do we have to handle inference model here?
    return models


def start_training(config, path_to_model=None, initial_epoch=None, split_name=None, inference_mode=False):
    tf.logging.set_verbosity(tf.logging.ERROR)
    is_dryrun = config.is_dryrun()
    
    print("\n{:-^80}".format("Preparing generator sequences for training"))
    training_sequence, validation_sequence = __get_sequences(config, split_name)
    
    print("\n{:-^80}".format("Preparing models for training"))
    training_model, inference_model = __get_model(config, path_to_model, initial_epoch, inference_mode, return_inference_model=True)
    
    print("\n{:-^80}".format("Preparing callbacks for training"))
    tagged_log_path, callbacks_listing = __get_callbacks(config, split_name)
    
    # TODO maybe implement as metric and use normal csv logger to log the values
    # but this is a bit differnt as a usual metric because we dont act on y_pred
    # well we act on y_pred but on the inference model
    if split_name == None:
        target_split = SPLIT_VALIDATE
    print("Loading ground truth captions for split:", "karpathy")
    ground_truth = to_caption_listing_by_image_id(load_prepared_captions_json_from_config(config, "karpathy"))
    prediction_sequence = MscocoPredictionSequence.create_mscoco_prediction_sequence_from_config(config, target_split)
    callbacks_listing.append(create_scores_logger(tagged_log_path,
                                                  inference_model,
                                                  prediction_sequence,
                                                  training_sequence.vocabulary,
                                                  ground_truth))
    callbacks_listing.append(EmbeddingProjectorCallback(logdir=tagged_log_path, layer_name="shatt_word_embeddings", vocabulary=training_sequence.vocabulary))
    callbacks_listing.append(TrainingScoresMerger(logdir=tagged_log_path))
    
    print("\n{:-^80}".format("Saving training configurations"))
    import os
    if not os.path.exists(tagged_log_path):
        os.makedirs(tagged_log_path)
        print("Create logging directory at " , tagged_log_path)
    else:    
        print("Logging directory already exists at " , tagged_log_path)    
    store_vocabulary(training_sequence.vocabulary, tagged_log_path, split_name)
    store_configuration(config, tagged_log_path, split_name)
    
    print("\n{:-^80}".format("Start training"))
    training_model.fit_generator(training_sequence,
                        validation_data=validation_sequence,
                        # validation_steps=10 if is_dryrun  else None, # is ignored when using sequences
                        # epochs=1 if is_dryrun else config.getEpochs(),
                        epochs=config.getEpochs(),
                        verbose=1,
                        # steps_per_epoch=10 if is_dryrun else None, # is ignored when using sequences
                        callbacks=callbacks_listing,
                        use_multiprocessing=config.getUseMultiProcessing(),
                        workers=config.getWorkers(),
                        max_queue_size=config.getMaxQueueSize(),
                        initial_epoch=initial_epoch if initial_epoch else 0)


import tensorflow as tf


def start_prediction(config, path_to_model, source_split, target_split):
    with tf.Session():
        # The following are loaded as 'flat' files on the top directory
        vocabulary = PaddingVocabulary.create_vocabulary_from_config(config, source_split)
        prediction_sequence = MscocoPredictionSequence.create_mscoco_prediction_sequence_from_config(config, target_split)
        
        model = __get_model(config, path_to_model, 1, inference_mode=True)
        print_model_summary(model, config)
        
        results = predict(model, prediction_sequence)
        results.write_results_file(vocabulary, path_to_model, target_split)
