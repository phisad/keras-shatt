import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from shatt.dataset.images.providers import MscocoFileSystemAttentionMapProvider, \
    MscocoFileSystemFeatureMapProvider, MscocoFileSystemImageProvider
from shatt.dataset.images import get_infix_from_config
from shatt.model import create_shatt_model_from_config
from shatt.dataset.vocabulary import Vocabulary
from shatt.dataset.results import MscocoCaptionResults
from shatt.dataset import store_json_to
import os

RESULT_FILE_PATTERN = "mscoco_caption_shatt_{}_results.json"

class ExperimentSequence(Sequence):
    
    def __init__(self, directory_path, image_prefix, caption_max_length, by_pass, target_shape=(448, 448), feature_size=196, return_scaled=True):
        print("Loading ATTENTION MAP provider at", directory_path)
        self.attention_provider = MscocoFileSystemAttentionMapProvider(directory_path, image_prefix)
        print("Amount of attention maps:", len(self.attention_provider))
        if by_pass:
            print("Loading IMAGE FEATURES provider at", directory_path)
            self.image_provider = MscocoFileSystemFeatureMapProvider(directory_path, image_prefix)
        else:
            print("Loading IMAGE provider at", directory_path)
            self.image_provider = MscocoFileSystemImageProvider(directory_path, image_prefix, vgg_like=True, target_size=target_shape)
        print("Amount of images:", len(self.image_provider))
        self.image_ids = self.attention_provider.get_image_ids()
        self.caption_max_length = caption_max_length
        self.standard_attention_map = np.ones((1, feature_size)) / feature_size
        self.return_scale = return_scaled

    def __to_list(self, list_or_array):
        list_or_array = np.array(list_or_array)
        list_or_array = np.squeeze(list_or_array)
        if len(np.shape(list_or_array)) == 0:
            list_or_array = np.asscalar(list_or_array)
            list_or_array = [list_or_array]
        else:
            list_or_array = list_or_array.tolist()
        list_or_array.append(0)
        return list_or_array

    def __getitem__(self, index):
        """Gets batch at position `index`.
    
          Arguments:
              index: position of the batch in the Sequence.
    
            Returns:
                A batch
        """
        image_id = self.image_ids[index]
        # print("getitem", image_id)
        
        attention_maps, attention_map_labels, attention_map_ids = self.attention_provider.get_attention_maps_for_image_id(image_id, return_scaled=self.return_scale)
        
        """ add sample with equal attention everywhere (category id: 0) """
        attention_maps = np.append(attention_maps, self.standard_attention_map, axis=0)
        attention_map_labels = self.__to_list(attention_map_labels)
        attention_map_ids = self.__to_list(attention_map_ids)
        
        """ amount of attention maps determines the batch size """ 
        batch_size = np.shape(attention_maps)[0]
        
        """ repeat the image for each box """
        images = self.image_provider.get_images_for_image_ids([image_id])
        images = np.repeat(images, batch_size, axis=0)
        
        """ mock the caption (not used in inference mode) """
        mock_captions = np.zeros(shape=(batch_size, self.caption_max_length, 1))
        # print(image_id, np.shape(images), np.shape(attention_maps), np.shape(attention_map_labels), np.shape(mock_captions))
        return {"input_captions" : mock_captions, "input_images" : images, "input_attention" : attention_maps}, ([image_id] * batch_size, attention_map_ids, attention_map_labels)
    
    def __len__(self):
        """Number of batch in the Sequence.
        
        Returns:
            The number of batches in the Sequence.
        """
        return len(self.image_ids)
    
    @staticmethod
    def create_from_config(config, split_name):
        directory_path = "/".join([config.getDatasetImagesDirectoryPath(), split_name])
        
        image_prefix = "COCO_" + get_infix_from_config(config, split_name)
        print("Image prefix:", image_prefix)
        
        caption_max_length = config.getCaptionMaximalLength()
        print("Caption max length:", caption_max_length)
        
        by_pass = config.getByPassImageFeaturesComputation()
        print("Bypass image feature compuration:", by_pass)
        
        target_shape = config.getImageInputShape()
        print("Target image shape:", target_shape)
        
        feature_size = config.getImageFeaturesSize()
        print("Image feature size:", feature_size)
        
        return ExperimentSequence(directory_path, image_prefix, caption_max_length, by_pass, target_shape, feature_size)

    
class ExperimentResults():
    
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.results = []
        
    def add_batch(self, batch_labels, batch_captions):
        """
            batch_labels: image ids, attention labels
            
            image ids are the same for each batch here
        """
        image_ids, box_ids, labels = batch_labels
        for idx, caption in enumerate(batch_captions):
            self.results.append((image_ids[idx], box_ids[idx], labels[idx], caption))
    
    def to_final_results(self):
        """
            Results Format
            results = [result]
            
            result {
                "image_id": int,
                "caption": str,
                "category": int,
                "box_id": int
            }
        """
        results = []
        for image_id, box_id, category, caption in self.results:
            result_caption = MscocoCaptionResults.to_result(caption, self.vocabulary)
            """ there are usually multiple boxes for each image """
            results.append({"box_id": str(box_id), "image_id" : str(image_id), "caption": result_caption, "category": str(category)})
        return results

    def write_results_file(self, target_path, file_infix):
        try:
            if os.path.isdir(target_path):
                directory_path = target_path
            else:
                directory_path = os.path.dirname(target_path)
                if directory_path == "":
                    directory_path = "."
            final_results = self.to_final_results()
            return store_json_to(final_results, directory_path, RESULT_FILE_PATTERN.format(file_infix))
        except Exception as e:
            print("Cannot write results file: " + str(e))
    
    @staticmethod
    def create_from_config(path_to_model):
        model_dir = os.path.dirname(path_to_model)
        vocabulary_path = model_dir + "/mscoco_vocabulary.json"
        vocabulary = Vocabulary.create_vocabulary_from_vocabulary_json(vocabulary_path, split_name=None, use_nltk=False)
        return ExperimentResults(vocabulary)

    