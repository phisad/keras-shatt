from shatt.dataset.images.providers import MscocoFileSystemFeatureMapProvider,\
    MscocoFileSystemImageProvider, ImageProvider
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
from shatt.dataset.images import get_image_paths, extract_to_image_ids_ordered,\
    to_image_path_by_id, load_numpy_from, _exists_image_path_by_id,\
    get_infix_from_config
from shatt.dataset.results import MscocoCaptionResults
from shatt.dataset.vocabulary import Vocabulary
import os
from shatt.dataset import store_json_to, load_json_from

RESULT_FILE_PATTERN = "mscoco_caption_shatt_{}_results.json"

class MscocoFileSystemExternalAttentionMapProvider(ImageProvider):
    """
        Given a batch of captions ids, the provider reads batches of attention maps from a directory on the file system.
        
        Notice: This might be slow during training, when the file system is the bottleneck. 
    """

    def __init__(self, directory_path=None, prefix="COCO_train2014"):
        super().__init__(directory_path, prefix)
    
    def __len__(self):
        return len(self.get_image_ids())
    
    def get_image_ids(self):
        paths = get_image_paths(self.directory_path, file_ending="bqx")
        return extract_to_image_ids_ordered(paths)
    
    def get_attention_maps_for_image_id(self, image_id):
        """
            attention_maps, question_infos, attention_types
        """
        attention_maps_per_type, questions_and_answers = self.get_images_for_image_ids([image_id])
        
        """ this is a single content list for a single image id """
        attention_maps_per_type = attention_maps_per_type[0]
        questions_and_answers = questions_and_answers[0] 
        
        attention_maps_listing = []
        attention_type_listing = []
        question_info_listing = []
        
        attention_types = ["word","phrase","question"]
        for tidx, attention_maps_per_question in enumerate(attention_maps_per_type):
            attention_type = attention_types[tidx]
            for qidx, attention_map in enumerate(attention_maps_per_question):
                attention_maps_listing.append(attention_map)
                attention_type_listing.append(attention_type)
                question_info = questions_and_answers["questions"][qidx]
                answer_info = questions_and_answers["answers"][qidx] 
                question_info_listing.append({**question_info, **answer_info})
        
        return np.array(attention_maps_listing), question_info_listing, attention_type_listing
    
    def get_images_for_image_ids(self, image_ids):
        """ returning a list because """
        attention_map_file_paths = [to_image_path_by_id(self.prefix, image_id, file_ending="bqx") for image_id in image_ids]
        attention_maps = [load_numpy_from(self.directory_path, image_file_path) for image_file_path in attention_map_file_paths]
        
        question_infos_file_paths = [to_image_path_by_id(self.prefix, image_id, file_ending="iqx") for image_id in image_ids]
        question_infos = [load_json_from(self.directory_path, image_file_path) for image_file_path in question_infos_file_paths]
        """ we cannot simply cast to numpy array here b.c. there might be a different amount of attention maps per image """
        """ what to do exactly (flatten, concatenate, sum) depends on the caller """ 
        return attention_maps, question_infos
    
    def _get_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        image_file_path = to_image_path_by_id(self.prefix, image_id, file_ending="bqx")
        return load_numpy_from(self.directory_path, image_file_path) 
    
    def _has_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        return _exists_image_path_by_id(self.prefix, image_id, self.directory_path, file_ending="bqx")

class ExternAttentionExperimentSequence(Sequence):
    """
        question_info       = {"questions":image_questions, "answers":answers} -> questions and answer sorted by sequential order
            answers         = [{ "answer": "pink and yellow",  "answer_type": "other" }]
            image_questions = [{ "image_id": 487025, "question": "Is there a shadow?","question_id": 4870251 }] 
    """
    
    def __init__(self, directory_path, image_prefix, caption_max_length, by_pass, target_shape=(448, 448), feature_size=196, return_scaled=True):
        print("Loading EXTERNAL ATTENTION MAP provider at", directory_path)
        self.attention_provider = MscocoFileSystemExternalAttentionMapProvider(directory_path, image_prefix)
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

    def __getitem__(self, index):
        """Gets batch at position `index`.
    
          Arguments:
              index: position of the batch in the Sequence.
    
            Returns:
                A batch
        """
        image_id = self.image_ids[index]
        
        attention_maps, question_infos, attention_types = self.attention_provider.get_attention_maps_for_image_id(image_id)
        
        """ amount of attention maps determines the batch size """ 
        batch_size = np.shape(attention_maps)[0]
        
        """ repeat the image for each attention map"""
        images = self.image_provider.get_images_for_image_ids([image_id])
        images = np.repeat(images, batch_size, axis=0)
        
        """ mock the caption (not used in inference mode) """
        mock_captions = np.zeros(shape=(batch_size, self.caption_max_length, 1))
        # print(image_id, np.shape(images), np.shape(attention_maps), np.shape(attention_map_labels), np.shape(mock_captions))
        return {"input_captions" : mock_captions, "input_images" : images, "input_attention" : attention_maps}, (question_infos, attention_types)
    
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
        
        return ExternAttentionExperimentSequence(directory_path, image_prefix, caption_max_length, by_pass, target_shape, feature_size)

    
class ExternAttentionExperimentResults():
    """
        question_info       = {"questions":image_questions, "answers":answers} -> questions and answer sorted by sequential order
            answers         = [{ "answer": "pink and yellow",  "answer_type": "other" }]
            image_questions = [{ "image_id": 487025, "question": "Is there a shadow?","question_id": 4870251 }] 
    """
    
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.results = []
        
    def add_batch(self, batch_labels, batch_captions):
        """
            batch_labels: question_infos, attention_types
        """
        question_infos, attention_types = batch_labels
        for idx, caption in enumerate(batch_captions):
            self.results.append((question_infos[idx], attention_types[idx], caption))
    
    def to_final_results(self):
        """
            Results Format
            results = [result]
            
            result {
                "image_id": int,
                "caption": str,     -> caption
                "category": int,    
                "box_id": int       -> type (of attention: control, word, phrase, question)
                                    -> question_info = {"question","answer"}
            }
        """
        results = []
        for question_info, attention_type, caption in self.results:
            result_caption = MscocoCaptionResults.to_result(caption, self.vocabulary)
            """ there are usually multiple attentions for each image """
            results.append({"caption": result_caption, "type": attention_type, "question_info": question_info})
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
        return ExternAttentionExperimentResults(vocabulary)

    