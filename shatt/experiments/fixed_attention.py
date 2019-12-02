'''
Created on 16.05.2019

@author: Philipp
'''
from shatt.model import create_shatt_model_from_config
from shatt.experiments import ExperimentSequence, ExperimentResults

RESULT_FIXED_ATTENTION_NAME = "attention_fixed"


class ExperimentModelBuilder():
    
    @staticmethod
    def create_from_config(config, path_to_model):
        model = create_shatt_model_from_config(config, inference_mode=True, use_input_attention=True)
        model.load_weights(path_to_model, by_name=True)
        return model


def start_fixed_attention(model, sequence, results, target_path):
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

            
def start_fixed_attention_from_config(config, path_to_model, target_split, target_path):
    print("\n{:-^80}".format("Create experiment sequence"))
    sequence = ExperimentSequence.create_from_config(config, target_split)
    print("\n{:-^80}".format("Create results handler"))
    results = ExperimentResults.create_from_config(path_to_model)
    print("\n{:-^80}".format("Load model"))
    model = ExperimentModelBuilder.create_from_config(config, path_to_model)
    print("\n{:-^80}".format("Start fixed attention experiment"))
    start_fixed_attention(model, sequence, results, target_path)
    print("\n{:-^80}".format("Experiment finished"))
    
