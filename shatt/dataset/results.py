'''
Created on 31.03.2019

@author: Philipp
'''

import os

from shatt.dataset import store_json_to
import numpy as np


class MscocoCaptionResults():
    
    def __init__(self):
        """ list questions-prediction tuples """
        self.results = []
        
    def add_batch(self, batch_image_ids, batch_captions):
        # prediction_classes = np.argmax(batch_predictions, axis=1)
        zipped = list(zip(batch_image_ids, batch_captions))
        self.results.extend(zipped)
    
    def get_captions_by_image_id_generator(self, vocabulary, amount):
        """ dont use this for scoring because <e> is probably not the end of the caption """
        for image_id, caption in self.results[:amount]:
            caption = np.squeeze(caption)
            # no break on zeros yet
            caption = [word for word in caption if word != 0]
            caption = vocabulary.encodings_to_captions([caption])[0]
            yield image_id, caption
    
    @staticmethod
    def to_result(caption, vocabulary):
        caption = np.squeeze(caption)
        result = []
        for w in caption:
            if w == vocabulary.get_end_symbol():
                break  # end symbol
            else:
                if w != 0:  # UNK
                    result.append(w)
        return vocabulary.encodings_to_captions([result])[0]
    
    def get_captions_by_image_id(self, vocabulary, return_multiple=False):
        results = {}
        for image_id, caption in self.results:
            result = MscocoCaptionResults.to_result(caption, vocabulary)
            if return_multiple:
                if image_id not in results:
                    results[image_id] = []
                results[image_id].append(result)
            else:
                results[image_id] = result
        return results

    def write_results_file(self, vocabulary, target_path, file_infix):
        """
            Results Format
            results = [result]
            
            result {
            "image_id": int,
            "caption": str
            }
        """
        try:
            if os.path.isdir(target_path):
                directory_path = target_path
            else:
                directory_path = os.path.dirname(target_path)
                if directory_path == "":
                    directory_path = "."
            results = [
                       {
                        "image_id" : str(image_id),
                        "caption" : predicted_caption
                       } 
                       for (image_id, predicted_caption) in self.get_captions_by_image_id(vocabulary).items()]
            return store_json_to(results, directory_path, "mscoco_caption_shatt_{}_results.json".format(file_infix))
        except Exception as e:
            print("Cannot write results file: " + str(e))
