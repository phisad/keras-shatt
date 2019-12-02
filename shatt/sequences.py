'''
Created on 14.03.2019

Utilities for the VQA 1.0 dataset
 
@author: Philipp
'''
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from shatt.dataset.images.providers import ImageProvider
import numpy as np
from shatt.dataset.captions import load_prepared_captions_json_from


class MsCocoCaptionSequence(Sequence):
    """
        The sequence to generate batches of (captions, image_provider) and (answers)
        
        This holds all possible captions and answers in memory, but loads the image_provider lazily.
    """
    
    def __init__(self, captions, image_provider, vocabulary, batch_size, dry_run=False):
        """
            @param captions: list
                The list of captions as dicts of { "caption", "image_id", "caption_id" }
            @param image_provider: ImageContainer
                The container to fetch the image_provider by caption dicts { "caption", "image_id", "caption_id" }
            @param vocabulary: Vocabulary
                The vocabulary to use to encode the textual captions
            @param batch_size: int
                The batch size to use
        """
        if batch_size > 300:
            raise Exception("Batch size shouldnt be more than 300 but is " + str(batch_size))
        self.captions = captions
        self.image_provider = image_provider
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.dry_run = dry_run
    
    def __len__(self):
        amount_of_batches = int(np.ceil(len(self.captions) / float(self.batch_size)))
        if self.dry_run and amount_of_batches > 10:
            return 10
        return amount_of_batches 
    
    def __getitem__(self, idx):
        batch_captions = self.captions[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._get_batch_for_input_captions(batch_captions)
    
    def _get_batch_for_input_captions(self, batch_captions):
        raise Exception("Not implemented")
    
    def one_shot_iterator(self):
        for idx in range(len(self)):
            yield self[idx]
    
    @staticmethod
    def create_mscoco_labelled_sequence_from_config(config, vocabulary, split_name):
        # The following are loaded as 'deep' having an own directory
        image_provider = ImageProvider.create_from_config(config, split_name)
        # The following are loaded as 'flat' files on the top directory
        prepared_captions = load_prepared_captions_json_from(config.getDatasetTextDirectoryPath(), split_name)
        return MscocoLabelledInputsSequence(prepared_captions, image_provider, vocabulary, config.getBatchSize(), config.getVocabularySize(), dry_run=config.is_dryrun())

        
class MscocoInputsSequence(MsCocoCaptionSequence):
    """
        The sequence to generate batches of (captions, images)
        
        This holds all possible captions and answers in memory, but loads the images on demand.
    """
    
    def __init__(self, captions, image_provider, vocabulary, batch_size, dry_run=False):
        super().__init__(captions, image_provider, vocabulary, batch_size, dry_run)

    def _get_batch_for_input_captions(self, batch_captions):
        """
            @return: a batch of inputs as a dict of input_captions and input_images
                        where input_captions are encoded and padded textual captions
                          and input_images are the according scaled image_provider
        """
        batch_textual_captions = [entry["caption"] for entry in batch_captions]
        batch_padded_captions = self.vocabulary.captions_to_encoding(batch_textual_captions, append_end_symbol=True)
        batch_padded_captions = np.expand_dims(batch_padded_captions, axis=-1)
        batch_images = self.image_provider.get_images_for_captions(batch_captions)
        return {"input_captions": batch_padded_captions, "input_images": batch_images}


class MscocoLabelledInputsSequence(MscocoInputsSequence):
    """
        The sequence to generate batches of (captions, image_provider) and (answers)
        
        This holds all possible captions and answers in memory, but loads the images on demand.
        
        @param num_classes: int
            The number of categories for the output labels
    """
    
    def __init__(self, captions, image_provider, vocabulary, batch_size, num_classes, dry_run=False):
        super().__init__(captions, image_provider, vocabulary, batch_size, dry_run)
        self.num_classes = num_classes + 1  # b.c. of padding value 0

    def _get_batch_for_input_captions(self, batch_captions):
        batch_captions_and_images = super()._get_batch_for_input_captions(batch_captions)
        batch_captions = batch_captions_and_images["input_captions"]
        batch_captions = np.squeeze(batch_captions)
        batch_one_hot_labels = to_categorical(batch_captions, self.num_classes)
        return (batch_captions_and_images, batch_one_hot_labels)
    

class MscocoPredictionSequence(Sequence):
    
    def __init__(self, image_provider, batch_size, caption_max_length, dry_run):
        self.image_provider = image_provider
        self.batch_size = batch_size
        self.dry_run = dry_run
        """ Use the image ids to choose from """
        self.image_ids = image_provider.get_image_ids()
        """ We provide the captions because the model is expecting """
        self.mock_captions = np.zeros(shape=(batch_size, caption_max_length, 1))
    
    def one_shot_iterator(self):
        for idx in range(len(self)):
            yield self[idx]
            
    def _get_batch_for_input_captions(self, batch_captions):
        batch_captions_and_images = super()._get_batch_for_input_captions(batch_captions)
        return (batch_captions_and_images, batch_captions)
    
    def __len__(self):
        amount_of_batches = int(np.ceil(len(self.image_ids) / float(self.batch_size)))
        if self.dry_run and amount_of_batches > 10:
            return 10
        return amount_of_batches 
    
    def __getitem__(self, idx):
        batch_image_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = self.image_provider.get_images_for_image_ids(batch_image_ids)
        return {"input_captions": self.mock_captions, "input_images": batch_images}, batch_image_ids
    
    @staticmethod
    def create_mscoco_prediction_sequence_from_config(config, target_split):
        # The following are loaded as 'deep' having an own directory
        image_provider = ImageProvider.create_from_config(config, target_split)
        return MscocoPredictionSequence(image_provider, config.getBatchSize(), config.getCaptionMaximalLength(), config.is_dryrun())
