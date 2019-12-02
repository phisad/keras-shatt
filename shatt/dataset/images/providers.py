'''
Created on 22.03.2019

@author: Philipp
'''

from tensorflow.keras.applications.vgg19 import preprocess_input
from shatt.dataset.images import to_image_path_by_id, load_numpy_from, \
    get_infix_from_config, _exists_image_path_by_id, get_image_paths, \
    extract_to_image_ids_ordered, draw_boxes_on_images, resize_images, \
    get_image, _get_image, softmax
import numpy as np
from shatt import SPLIT_VALIDATE, SPLIT_TRAINVAL, SPLIT_TRAIN, SPLIT_TEST_DEV, \
    SPLIT_TEST
import os
import collections


class ImageProvider():
    """
        Given a batch of captions ids, the provider returns a batches of corresponding images.
    """

    def __init__(self, directory_path=None, prefix="COCO_train2014"):
        """
            @param directory_path: str
                Directory containing the image_provider with names of COCO_train2014_<12-digit-image-id> for example COCO_train2014_000000000009
            @param prefix: str
                The image file name prefixed to the image id part
        """
        self.directory_path = directory_path
        self.prefix = prefix
        
    def __len__(self):
        return len(os.listdir(self.directory_path))
    
    def get_image_ids(self):
        raise Exception("Not implemented")
    
    def get_images_for_captions(self, captions):
        image_ids = [entry["image_id"] for entry in captions]
        return self.get_images_for_image_ids(image_ids)

    def get_images_for_image_ids(self, image_ids):
        raise Exception("Not implemented")
    
    def _get_image_for_image_id(self, image_id):
        raise Exception("Not implemented")
    
    def _has_image_for_image_id(self, image_id):
        raise Exception("Not implemented")
    
    @staticmethod
    def create_multi_split_provider_from_config(config, split_names):
        providers = [ImageProvider.create_single_split_provider_from_config(config, split_name) for split_name in split_names]
        return MscocoMultipleExclusiveProvider(providers)

    @staticmethod
    def create_single_split_provider_from_config(config, split_name):
        directory_path = "/".join([config.getDatasetImagesDirectoryPath(), split_name])
        image_infix = get_infix_from_config(config, split_name)
        if config.getByPassImageFeaturesComputation():
            return MscocoFileSystemFeatureMapProvider(directory_path, prefix="COCO_" + image_infix)
        else:
            return MscocoFileSystemImageProvider(directory_path, prefix="COCO_" + image_infix, vgg_like=True, target_size=config.getImageInputShape())    
        
    @staticmethod
    def create_from_config(config, split_name):
        if split_name == SPLIT_TRAINVAL:
            print("Identified trainval split for image provider")
            return ImageProvider.create_multi_split_provider_from_config(config, [SPLIT_TRAIN, SPLIT_VALIDATE])
        elif split_name == SPLIT_TEST_DEV:
            return ImageProvider.create_single_split_provider_from_config(config, SPLIT_TEST)
        elif split_name == "karpathy":
            return ImageProvider.create_single_split_provider_from_config(config, SPLIT_VALIDATE)
        else:
            return ImageProvider.create_single_split_provider_from_config(config, split_name)


class MscocoMultipleExclusiveProvider(ImageProvider):
    """
        This provider is useful, when there are multiple existing directories for images. Then this provider manages the retrieval from these.
        
        Given a batch of captions ids, the provider reads batches of feature maps from a directory on the file system.
        
        Here, the given providers are asked if they can succeed the request. Then the first accepting provider will be used to complete the request.
        
        Notice: This might be slow during training, when the file system is the bottleneck. 
    """

    def __init__(self, providers):
        super().__init__(None, None)
        self.providers = providers
        
    def _get_image_for_image_id(self, image_id):
        for provider in self.providers:
            if provider._has_image_for_image_id(image_id):
                return provider._get_image_for_image_id(image_id)
        raise Exception("Cannot serve image for image id '{}'".format(image_id))
    
    def __len__(self):
        return sum([len(provider) for provider in self.providers])
    
    def get_image_ids(self):
        results = []
        for provider in self.providers:
            results.extend(provider.get_image_ids())
        return results
    
    def get_images_for_image_ids(self, image_ids):
        """
            @return: the images in the same order as the image ids
        """
        images = np.array([self._get_image_for_image_id(image_id) for image_id in image_ids]) 
        return images

    
class MscocoFileSystemFeatureMapProvider(ImageProvider):
    """
        Given a batch of captions ids, the provider reads batches of feature maps from a directory on the file system.
        
        Notice: This might be slow during training, when the file system is the bottleneck. 
    """

    def __init__(self, directory_path=None, prefix="COCO_train2014"):
        super().__init__(directory_path, prefix)
    
    def __len__(self):
        return len(self.get_image_ids())
    
    def get_image_ids(self):
        paths = get_image_paths(self.directory_path, file_ending="npy")
        return extract_to_image_ids_ordered(paths)
        
    def get_images_for_image_ids(self, image_ids):
        """
            @param captions: the list of captions as dicts of { "caption", "image_id", "id" }
            
            @return: the image in the same order as captions
        """
        feature_map_file_paths = [to_image_path_by_id(self.prefix, image_id, file_ending="npy") for image_id in image_ids]
        feature_maps = np.array([load_numpy_from(self.directory_path, image_file_path) for image_file_path in feature_map_file_paths])
        return feature_maps
    
    def _get_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        image_file_path = to_image_path_by_id(self.prefix, image_id, file_ending="npy")
        return load_numpy_from(self.directory_path, image_file_path) 
    
    def _has_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        return _exists_image_path_by_id(self.prefix, image_id, self.directory_path, file_ending="npy")


class MscocoFileSystemAttentionMapProvider(ImageProvider):
    """
        Given a batch of captions ids, the provider reads batches of attention maps from a directory on the file system.
        
        Notice: This might be slow during training, when the file system is the bottleneck. 
    """

    def __init__(self, directory_path=None, prefix="COCO_train2014"):
        super().__init__(directory_path, prefix)
    
    def __len__(self):
        return len(self.get_image_ids())
    
    def get_image_ids(self):
        paths = get_image_paths(self.directory_path, file_ending="bbx")
        return extract_to_image_ids_ordered(paths)
    
    def get_attention_maps_for_image_id(self, image_id, return_scaled=False, return_mask=False):
        attention_maps, attention_labels, attention_ids = self.get_images_for_image_ids([image_id])
        attention_maps = np.array(attention_maps)
        attention_maps = np.squeeze(attention_maps)
        """ the downsizing might introduce fragments """
        if len(np.shape(attention_maps)) == 1:
            attention_maps = np.expand_dims(attention_maps, axis=0) 
        if return_mask:
            #[print(a) for a in attention_maps]
            #attention_maps = np.array([np.where(a < np.mean(a), 0, 1) for a in attention_maps])
            attention_maps = np.array([np.where(a < np.mean(a), 0, 255) for a in attention_maps])
        if return_scaled:
            attention_maps = np.interp(attention_maps, (attention_maps.min(), attention_maps.max()), (0, 1))
            attention_maps = softmax(attention_maps, axis=1)
        return attention_maps, attention_labels, attention_ids
    
    def get_images_for_image_ids(self, image_ids):
        """ returning a list because """
        attention_map_file_paths = [to_image_path_by_id(self.prefix, image_id, file_ending="bbx") for image_id in image_ids]
        attention_maps = [load_numpy_from(self.directory_path, image_file_path) for image_file_path in attention_map_file_paths]
        
        attention_labels_file_paths = [to_image_path_by_id(self.prefix, image_id, file_ending="lbx") for image_id in image_ids]
        attention_labels = [load_numpy_from(self.directory_path, image_file_path) for image_file_path in attention_labels_file_paths]
        
        attention_ids_file_paths = [to_image_path_by_id(self.prefix, image_id, file_ending="ibx") for image_id in image_ids]
        attention_ids = [load_numpy_from(self.directory_path, image_file_path) for image_file_path in attention_ids_file_paths]
        """ we cannot simply cast to numpy array here b.c. there might be a different amount of attention maps per image """
        """ what to do exactly (flatten, concatenate, sum) depends on the caller """ 
        return attention_maps, attention_labels, attention_ids
    
    def _get_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        image_file_path = to_image_path_by_id(self.prefix, image_id, file_ending="bbx")
        return load_numpy_from(self.directory_path, image_file_path) 
    
    def _has_image_for_image_id(self, image_id):
        """ Overwritten to determine the file ending """
        return _exists_image_path_by_id(self.prefix, image_id, self.directory_path, file_ending="bbx")

        
class MscocoFileSystemImageProvider(ImageProvider):
    """
        Given a batch of captions ids, the provider reads batches of images from a directory on the file system.
        
        The images are preprocessed on the fly like in image preparation:
        - reduce image size to target size
        - apply vgg like preprocessing if applicable
        
        Notice: This might be slow during training, when the file system is the bottleneck. 
    """
    
    def __init__(self, directory_path=None, prefix="COCO_train2014", vgg_like=True, target_size=(448, 448)):
        """
            @param vgg_like: bool
                When true, then the image_provider are prepared for VGG use according to the 'mode'. 
                
                mode: One of "caffe", "tf" or "torch".
                    - caffe: will convert the image_provider from RGB to BGR,
                        then will zero-center each color channel with
                        respect to the ImageNet dataset,
                        without scaling.
                    - tf: will scale pixels between -1 and 1, sample-wise.
                    - torch: will scale pixels between 0 and 1 and then
                        will normalize each channel with respect to the
                        ImageNet dataset.
                        
                We use mode 'tf' because thats the backend here.
        """
        super().__init__(directory_path, prefix)
        self.vgg_like = vgg_like
        if target_size and len(target_size) == 3:
            target_size = (target_size[0], target_size[1])
        self.target_size = target_size
        
    def __len__(self):
        return len(self.get_image_ids())
    
    def get_image_ids(self):
        paths = get_image_paths(self.directory_path)
        return extract_to_image_ids_ordered(paths)
    
    def get_images_for_image_ids(self, image_ids):
        """
            @param captions: the list of captions as dicts of { "caption", "image_id", "id" }
            
            @return: the image in the same order as captions
        """
        image_file_paths = [to_image_path_by_id(self.prefix, image_id, self.directory_path) for image_id in image_ids]
        images = np.array([_get_image(image_file_path, self.target_size) for image_file_path in image_file_paths])
        if self.vgg_like:
            images = preprocess_input(images, mode="caffe")
        return images


class MscocoFileSystemBoundingBoxesDrawingImageProvider(ImageProvider):
    """ Actually never used during training """
    
    def __init__(self, bounding_boxes, directory_path=None, prefix="COCO_train2014", target_size=(448, 448)):
        super().__init__(directory_path, prefix)
        if target_size and len(target_size) == 3:
            target_size = (target_size[0], target_size[1])
        self.target_size = target_size
        self.boxes_by_id = collections.defaultdict(list)
        [self.boxes_by_id[box["image_id"]].append(box) for box in bounding_boxes]
        
    def __len__(self):
        return len(self.get_image_ids())
    
    def get_image_ids(self):
        paths = get_image_paths(self.directory_path)
        return extract_to_image_ids_ordered(paths)
    
    def get_boxes_by_image_ids(self, image_ids):
        return [self.boxes_by_id[image_id] for image_id in image_ids]
    
    def get_images_for_image_ids(self, image_ids):
        """
            @param captions: the list of captions as dicts of { "caption", "image_id", "id" }
            
            @return: the image in the same order as captions
        """
        image_file_paths = [to_image_path_by_id(self.prefix, image_id, self.directory_path) for image_id in image_ids]
        """ original sized images b.c. boxes are from there """
        images = [get_image(image_file_path) for image_file_path in image_file_paths]
        image_boxes = self.get_boxes_by_image_ids(image_ids)
        images = draw_boxes_on_images(images, image_boxes)
        images = resize_images(images, self.target_size)
        return images, image_boxes
