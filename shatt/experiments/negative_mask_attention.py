'''
Created on 16.05.2019

@author: Philipp
'''
from shatt.model import create_shatt_model_from_config
from shatt.experiments import ExperimentSequence, ExperimentResults
from tensorflow.python.keras.layers import Input, Lambda
import tensorflow as tf

RESULT_FIXED_ATTENTION_NAME = "attention_dynamic"


class ExperimentModelBuilder():
    
    @staticmethod
    def create_from_config(config, path_to_model, input_weight=1):
        def dynamic_sub(x):
            current_attention = x[0]
            input_attention = x[1]
            attention = (current_attention - input_weight * input_attention) 
            attention = tf.keras.backend.clip(attention, 0, 1)
            return attention
    
        input_attention = Input(shape=(196,), name="input_attention")
        output_attention = Lambda(dynamic_sub)
        
        model = create_shatt_model_from_config(config, inference_mode=True, dynamic_attention=True, attention_graph=(output_attention, input_attention))
        model.load_weights(path_to_model, by_name=True)
        return model


def start_negative_mask(model, sequence, results, target_path):
        processed_count = 0
        expected_num_batches = len(sequence)
        try:
            for batch_inputs, batch_labels in sequence:
                batch_predictions = model.predict_on_batch(batch_inputs)
                results.add_batch(batch_labels, batch_predictions)
                processed_count = processed_count + 1
                print(">> Processing batches {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
            print("\n{:-^80}".format("Writing results"))
            results.write_results_file(target_path, file_infix=RESULT_FIXED_ATTENTION_NAME)
        except Exception as e:
            print("Exception: ", e)

            
def start_negative_mask_attention_from_config(config, path_to_model, target_split, target_path, input_weight):
    print("\n{:-^80}".format("Create experiment sequence"))
    sequence = ExperimentSequence.create_from_config(config, target_split, return_scaled=False)
    print("\n{:-^80}".format("Create results handler"))
    results = ExperimentResults.create_from_config(path_to_model)
    print("\n{:-^80}".format("Load model"))
    model = ExperimentModelBuilder.create_from_config(config, path_to_model, input_weight)
    print("\n{:-^80}".format("Start dynamic attention experiment"))
    start_negative_mask(model, sequence, results, target_path)
    print("\n{:-^80}".format("Experiment finished"))
    
