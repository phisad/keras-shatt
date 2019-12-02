'''
Created on 10.05.2019

@author: Philipp
'''
import tensorflow as tf
import time
import os
import sys
from shatt.dataset.results import MscocoCaptionResults
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from shatt.scorer.bleu.bleu import Bleu
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.engine.input_layer import Input
from shatt.dataset import store_tsv_to
from tensorflow.python.layers.core import Flatten

        
def predict(model, sequence):
    results = MscocoCaptionResults()        
    processed_count = 0
    expected_num_batches = len(sequence)
    try:
        for batch_inputs, batch_image_ids in sequence.one_shot_iterator():
            batch_predictions = model.predict_on_batch(batch_inputs)
            results.add_batch(batch_image_ids, batch_predictions)
            processed_count = processed_count + 1
            print(">> Processing batches {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
    except Exception as e:
        print("Exception: ", e)
    return results


class EmbeddingProjectorCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, logdir, layer_name, vocabulary, batch_size=100):
        self.layer_name = layer_name
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.logdir = logdir
        self.meta_filename = "metadata.tsv"
        
    def on_train_begin(self, logs):
        """ establish embedding model """
        layer = self.model.get_layer(self.layer_name)
        if layer == None:
            raise Exception("Cannot find layer with name " + self.layer_name)
        input_words = Input(shape=(1,), name="embedding_callback_input_words")
        layer_output = layer(input_words)
        layer_output = Flatten(name="embedding_callback_flatten")(layer_output)
        self.embedding_model = Model(inputs=input_words, outputs=layer_output)
        
        """ write metadata.tsv """ 
        word_sequence = self.vocabulary.get_word_sequence(padding_symbol="<PAD>")
        metadata = [{"Word":w, "Frequency": self.vocabulary.get_word_count(w)} for w in word_sequence]
        store_tsv_to(metadata, self.logdir, self.meta_filename)
        
        """ encode sequence""" 
        self.encoded_word_sequence = self.vocabulary.get_encoded_word_sequence(include_padding=True)
        
    def __get_num_batches(self):
        return int(np.ceil(len(self.encoded_word_sequence) / float(self.batch_size))) 
    
    def __get_batch(self, idx):
        return self.encoded_word_sequence[idx * self.batch_size:(idx + 1) * self.batch_size]
    
    def one_shot_iterator(self):
        for idx in range(self.__get_num_batches()):
            yield self.__get_batch(idx)
            
    def on_epoch_end(self, epoch, logs):
        """ Calculate word embeddings using keras """
        processed_count = 0
        expected_num_batches = self.__get_num_batches()
        results = []
        try:
            for words in self.one_shot_iterator():
                words = np.expand_dims(words, axis=-1)
                word_embeddings = self.embedding_model.predict_on_batch(words)
                results.extend(word_embeddings)
                processed_count = processed_count + 1
                print(">> Computing word embeddings {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
        except Exception as e:
            print("Exception: ", e)
        """
            Save embedding projection using tensorflow. Thus we can also use a decoupled graph (if necessary).
            
            The results are only the embedded words. These embedding vectors are then saved.
            The projector is taking the vector and they are aligned with metadata.tsv
            Therefore the outputs order must match the metadata.tsv order
        """
        results = np.array(results)
        variable_name = 'word_embeddings_epoch_{}'.format(epoch)
        checkpoint_name = "/".join([self.logdir, "{}.ckpt".format(variable_name)])
        results_variable = tf.Variable(results, name=variable_name)
         
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = results_variable.name
        embedding.metadata_path = "./metadata.tsv"
        
        try:
            init_op = tf.variables_initializer([results_variable])
            with tf.Session() as sess:
                sess.run(init_op)
                # write checkpoint to logdir
                saver = tf.train.Saver({variable_name: results_variable})
                saver.save(sess, checkpoint_name)
                # write projector_config.pbtxt
                summary_writer = tf.summary.FileWriter(self.logdir)
                projector.visualize_embeddings(summary_writer, config)
        except:
            err_msg = sys.exc_info()[0]
            print("Could not save word embeddings because " + str(err_msg))

        """ remove old checkpoints if possible """
        checkpoint_files = [file for file in os.listdir(self.logdir) if "ckpt" in file]
        checkpoint_files = [file for file in checkpoint_files if str(epoch) not in file]
        for file in checkpoint_files:
            try:
                os.remove("/".join([self.logdir, file]))
            except:
                err_msg = sys.exc_info()[0]
                print("Could not remove old word embeddings file {} because {}".format(file, str(err_msg)))


def evaluate_bleu_on_results(results, correct_captions, vocabulary):
        """ notice: here we call indeed MscocoCaptionResults.to_result so that <e> is the end tag """
        generated_captions = results.get_captions_by_image_id(vocabulary)
        return evaluate_bleu_scores_and_print(correct_captions, generated_captions)
        
def evaluate_bleu_scores_and_print(correct_captions, generated_captions):
        final_scores = evaluate_bleu_scores(correct_captions, generated_captions)
        """ target scores as in the original paper """
        for score_name, target_score in [("Bleu_1", 71.8),
                                         ("Bleu_2", 49.2),
                                         ("Bleu_3", 34.4),
                                         ("Bleu_4", 24.3)]:
            print("{}: {:.1f} / {:.1f} ({:.1f})".format(score_name, final_scores[score_name], target_score, target_score - final_scores[score_name]))  
        return final_scores


def evaluate_bleu_scores(ground_truth, generate_captions):
    """
        @param ground_truth: image_id by list with multiple ground truth captions
        @param generate_captions: image_id by list with single caption
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        # (Rouge(), "ROUGE_L"),
        # (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(ground_truth, generate_captions)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = np.round(s * 100, 1)
        else:
            final_scores[method] = np.round(score * 100, 1)
    return final_scores        
    
class ScoresCallback(tf.keras.callbacks.CSVLogger):
    
    def __init__(self , filename, inference_model, prediction_sequence, vocabulary, ground_truth):
        super().__init__(filename)
        self.inference_model = inference_model
        self.prediction_sequence = prediction_sequence
        self.vocabulary = vocabulary
        self.ground_truth = ground_truth

    def on_epoch_end(self, epoch, logs=None):
        results = predict(self.inference_model, self.prediction_sequence)
        results.write_results_file(self.vocabulary, self.filename, "validate_epoch_{:03}".format(epoch + 1))
        
        """ notice: here we dont call MscocoCaptionResults.to_result but this is not used anywhere else """
        [print(captiont) for captiont in results.get_captions_by_image_id_generator(self.vocabulary, 9)]
        
        final_scores = evaluate_bleu_on_results(results, self.ground_truth, self.vocabulary)
        
        super().on_epoch_end(epoch, logs=final_scores)
    
def create_scores_logger(tagged_log_path, inference_model, prediction_sequence, vocabulary, ground_truth):
    logname = "/".join([tagged_log_path, "scores.csv"])
    print("- Scores log: " + logname)
    return ScoresCallback(logname, inference_model, prediction_sequence, vocabulary, ground_truth)


def create_reduce_lr_on_plateau(callbacks_listing):
    """ Thi is propably not necessary when using adam optimizer """
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                                monitor="categorical_accuracy",
                                factor=0.5, patience=1, verbose=1,
                                mode="max")
    callbacks_listing.append(reduce_lr)
    print("Added lr reduction after epochs without improvement")


def create_early_stopping():
    early_stopping = tf.keras.callbacks.EarlyStopping(
                                monitor="categorical_accuracy",
                                min_delta=0.0001, patience=5, verbose=1,
                                mode="max", baseline=0.25)
    print("- Early stopping monitor: after 5 epochs [categorical_accuracy]")
    return early_stopping


def create_tensorboard_from_save_path(base_path, model_log_path, update_frequency):
    """
        Create tensorboard callback based on a given save_path. 
        The dirname from the given path is used to construct the log path.
    """
    time_tag = time.strftime("%H-%M-%S", time.gmtime())
    
    tagged_log_path = "{}/{}/{}".format(base_path, model_log_path, time_tag)
    print("- Tensorboard log: " + tagged_log_path)
    
    """ The order of lines in the metadata file is assumed to match the order of vectors in the embedding variable """
    tensorboard_logger = tf.keras.callbacks.TensorBoard(log_dir=tagged_log_path, write_graph=True)
    return tagged_log_path, tensorboard_logger


def create_csv_logger(tagged_log_path):
    logname = "/".join([tagged_log_path, "training.csv"])
    print("- CSV log: " + logname)
    csvlogger = tf.keras.callbacks.CSVLogger(logname)
    return csvlogger


def create_checkpointer(tagged_log_path, model_name, store_per_epoch=False):
    """
        Create checkpoint callback based on a given log_path. 
    """
    if store_per_epoch:
        model_name = model_name + ".{epoch:03d}.h5"
    else:
        model_name = model_name + ".h5"
    model_path = "/".join([tagged_log_path, model_name])
    print("- Checkpoint monitor: max [masked_categorical_accuracy] at " + model_path)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, monitor="masked_categorical_accuracy", mode="max",
                                                      save_best_only=False, save_weights_only=True, verbose=1)
    return checkpointer


class TrainingScoresMerger(tf.keras.callbacks.Callback):
    
    def __init__(self, logdir):
        self.logdir = logdir
        
    def on_epoch_end(self, epoch, logs=None):
        merge_training_scores_csv(self.logdir)

    
def merge_training_scores_csv(model_dir):
    import csv
    
    training_csv_path = model_dir + "/training.csv"
    training_csv = []
    with open(training_csv_path) as f:
        reader = csv.DictReader(f, fieldnames=["epoch", "loss", "masked_categorical_accuracy", "val_loss", "val_masked_categorical_accuracy"])
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            line["loss"] = np.round(float(line["loss"]), 2)
            line["val_loss"] = np.round(float(line["val_loss"]), 2)
            
            line["masked_categorical_accuracy"] = float(line["masked_categorical_accuracy"])
            line["val_masked_categorical_accuracy"] = float(line["val_masked_categorical_accuracy"])
            
            line["masked_categorical_accuracy"] = np.round(line["masked_categorical_accuracy"], 2)
            line["val_masked_categorical_accuracy"] = np.round(line["val_masked_categorical_accuracy"], 2)
            
            line["mca"] = line.pop("masked_categorical_accuracy")
            line["val_mca"] = line.pop("val_masked_categorical_accuracy")
            
            training_csv.append(line)
    training_csv_by_id = dict([(int(t["epoch"]), t) for t in training_csv])
    
    scores_csv_path = model_dir + "/scores.csv"
    scores_csv = []
    with open(scores_csv_path) as f:
        reader = csv.DictReader(f, fieldnames=["epoch", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            scores_csv.append(line)
    scores_csv_by_id = dict([(int(t["epoch"]), t) for t in scores_csv])
    
    merge_csv = []
    for epoch, t in training_csv_by_id.items():
        s = scores_csv_by_id[epoch]
        z = {**s, **t}
        merge_csv.append(z)
    
    merge_csv = sorted(merge_csv, key=lambda x : (x["Bleu_4"], x["Bleu_3"], x["Bleu_2"], x["Bleu_1"]), reverse=True)
    path = model_dir + "/training_scores.csv"
    with open(path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "mca", "val_mca", "loss", "val_loss"])
        writer.writeheader()
        writer.writerows(merge_csv)
