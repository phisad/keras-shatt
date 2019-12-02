'''
Created on 10.05.2019

@author: Philipp
'''
from shatt.dataset import load_json_from, store_json_to
from shatt import SPLIT_TRAINVAL, SPLIT_TRAIN, SPLIT_VALIDATE
import collections
import numpy as np

DEFAULT_CATEGORIES_FILE_NAME = "mscoco_boxes_categories.json"

DEFAULT_CATEGORIES_SPLIT_FILE_NAME_PATTERN = "mscoco_boxes_categories_{}.json"

DEFAULT_PREPARED_BOXES_FILE_NAME = "mscoco_boxes.json"

DEFAULT_PREPARED_BOXES_SPLIT_FILE_NAME_PATTERN = "mscoco_boxes_{}.json"

DEFAULT_BOXES_FILE_NAME = "instances.json"

def calculate_metrics(boxes_json):
    sizes = []
    for entry in boxes_json:
        _, _, width, height = entry["box"] 
        sizes.append([width, height])
    sizes = np.array(sizes)
    mean = np.mean(sizes, axis=0)
    median = np.median(sizes, axis=0)
    print("mean width and height", mean)
    print("median width and height", median)
    
    # get indices where box widht and height are greater equal the median
    greaterequal = np.flatnonzero(np.all(np.greater_equal(sizes, median), axis=1))
    greater_set = set(greaterequal)
    boxes = [box for idx, box in enumerate(boxes_json) if idx in greater_set]
    print("total", "amount of boxes", len(sizes))
    print("(width, height) >= median", "amount of boxes", len(boxes))
    return boxes
        
def to_boxes_by_id(boxes_json):
    boxes_by_id = collections.defaultdict(list)
    [boxes_by_id[box["image_id"]].append(box) for box in boxes_json]
    return boxes_by_id

        
def load_categories_json_from(directory_path_or_file, split_name=None, flat=True):
    """
        [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person"
            },
        ]
    """
    lookup_filename = DEFAULT_CATEGORIES_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        lookup_filename = DEFAULT_CATEGORIES_SPLIT_FILE_NAME_PATTERN.format(split_name) 
        
    return load_json_from(directory_path_or_file, lookup_filename)


def load_categories_by_id(directory_path_or_file, split_name=None, flat=True):
    """ 
        return the categories in the correct order
        meaning that the category 1 is at index - 1 
    """
    categories = load_categories_json_from(directory_path_or_file, split_name, flat)
    categories = dict([(category["id"], category) for category in categories])
    return categories


def load_textual_categories_by_id(directory_path_or_file, split_name=None, flat=True):
    """ 
        return the categories in the correct order
        meaning that the category 1 is at index - 1 
    """
    categories = load_categories_json_from(directory_path_or_file, split_name, flat)
    categories = dict([(category["id"], category["name"]) for category in categories])
    return categories


def load_instances_json_from(directory_path_or_file, split_name=None, flat=False):
    """
        @param split_name: when given looks for the sub-directory or file in the flat directory
        @param flat: when True looks for a file in the given directory, otherwise looks into the sub-directory 
    """
    lookup_filename = DEFAULT_BOXES_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        raise Exception("Not supported to have source instances files on the same level as the top dataset directory")
        
    return load_json_from(directory_path_or_file, lookup_filename)


def load_prepared_boxes_json_from_config(config, split_name):
    return load_prepared_boxes_json_from(config.getDatasetTextDirectoryPath(), split_name)


def load_prepared_boxes_json_from(directory_path_or_file, split_name=None, flat=True):
    lookup_filename = DEFAULT_PREPARED_BOXES_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        lookup_filename = DEFAULT_PREPARED_BOXES_SPLIT_FILE_NAME_PATTERN.format(split_name) 
        
    return load_json_from(directory_path_or_file, lookup_filename)


def store_categories_as_file(categories, target_directory_path_or_file, split_name=None):
    lookup_filename = DEFAULT_CATEGORIES_FILE_NAME
    if split_name:    
        lookup_filename = DEFAULT_CATEGORIES_SPLIT_FILE_NAME_PATTERN.format(split_name) 
    return store_json_to(categories, target_directory_path_or_file, lookup_filename)

    
def store_prepared_boxes_as_file(boxes, target_directory_path_or_file, split_name=None):
    lookup_filename = DEFAULT_PREPARED_BOXES_FILE_NAME
    if split_name:    
        lookup_filename = DEFAULT_PREPARED_BOXES_SPLIT_FILE_NAME_PATTERN.format(split_name) 
    return store_json_to(boxes, target_directory_path_or_file, lookup_filename)


def create_prepared_boxes_file(config, split_name):
    """ 
        we create a single boxes.json file
    """
    directory_path = config.getDatasetTextDirectoryPath()
    if split_name == SPLIT_TRAINVAL:
        training_captions_json = load_instances_json_from(directory_path, SPLIT_TRAIN)
        valdiation_captions_json = load_instances_json_from(directory_path, SPLIT_VALIDATE)
        instances = training_captions_json["annotations"] + valdiation_captions_json["annotations"]
        categories = training_captions_json["categories"] + valdiation_captions_json["categories"]
    else:
        # use training labels also for the validate split to filter the captions
        instances_json = load_instances_json_from(directory_path, split_name)
        instances = instances_json["annotations"]
        categories = instances_json["categories"]
        """
            A instance is an entry like
            {
                "id", int,
                "image_id": int,
                "bbox": [x,y,width,height]
                "category_id": int
            }
        """
    store_categories_as_file(categories, directory_path, split_name)
    
    print("Create prepared boxes file")
    prepared_boxes = [
            {
                "box": annotation["bbox"],
                "box_id": annotation["id"],
                "category_id": annotation["category_id"],
                "image_id": annotation["image_id"],
            }
        for annotation in instances]
    
    return store_prepared_boxes_as_file(prepared_boxes, directory_path, split_name)

