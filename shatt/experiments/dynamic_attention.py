'''
Created on 16.05.2019

@author: Philipp
'''
from shatt.model import create_shatt_model_from_config
from shatt.experiments import ExperimentSequence, ExperimentResults
from tensorflow.python.keras.layers import Input, Lambda

RESULT_FIXED_ATTENTION_NAME = "attention_dynamic"


class ExperimentModelBuilder():
    
    @staticmethod
    def create_from_config(config, path_to_model, input_weight=1):
        def dynamic_add(x):
            current_attention = x[0]
            input_attention = x[1]
            attention = (current_attention + input_weight * input_attention) / (input_weight + 1)
            return attention
    
        input_attention = Input(shape=(196,), name="input_attention")
        output_attention = Lambda(dynamic_add)
        
        model = create_shatt_model_from_config(config, inference_mode=True, dynamic_attention=True, attention_graph=(output_attention, input_attention))
        model.load_weights(path_to_model, by_name=True)
        return model


def start_dynamic_attention(model, sequence, results, target_path):
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

            
def start_dynamic_attention_from_config(config, path_to_model, target_split, target_path, input_weight):
    print("\n{:-^80}".format("Create experiment sequence"))
    sequence = ExperimentSequence.create_from_config(config, target_split)
    print("\n{:-^80}".format("Create results handler"))
    results = ExperimentResults.create_from_config(path_to_model)
    print("\n{:-^80}".format("Load model"))
    model = ExperimentModelBuilder.create_from_config(config, path_to_model, input_weight)
    print("\n{:-^80}".format("Start dynamic attention experiment"))
    start_dynamic_attention(model, sequence, results, target_path)
    print("\n{:-^80}".format("Experiment finished"))
    
