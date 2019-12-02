'''
Created on 22.03.2019

    Creates many numpy files which contain a feature map array each. The file name is indicating the image id.
    
    - The image id and preprocessed images are loaded from a TFRecord file
    - The image ids are gathered from the file paths
    - The image ids and feature maps are jointly stored to a single file (expecting the order is kept)
    
    This workflow is a good option, when the data does not fit into memory.
    
@author: Philipp
'''
from shatt.dataset.images import load_numpy_from, to_image_path_by_id, store_numpy_to, get_infix_from_config, get_image_paths, draw_box_on_image
from shatt.dataset.boxes import load_prepared_boxes_json_from_config,\
    calculate_metrics
from shatt.dataset import to_split_dir

import sys

from multiprocessing import Pool
import tqdm
from tensorflow.keras.preprocessing.image import load_img
from shatt.dataset.images import extract_image_id
import collections
import os
from PIL import Image, ImageDraw
import numpy as np
from PIL.Image import LANCZOS, NEAREST, BILINEAR


def load_attention_map_by_image_id_from_many(image_ids, directory_path, image_prefix, split_name=None):
    """
        Returns a dict of {image_id : feature_map} for each given image_id.
    """
    if split_name:
        directory_path = "/".join([directory_path, split_name])
    return dict([(image_id, load_numpy_from(to_image_path_by_id(image_prefix, image_id, directory_path, file_ending="bbx"))) 
                for image_id in image_ids])


def __get_batch_size(config, run_opts):
    if run_opts["batch_size"]:
        return run_opts["batch_size"]
    batch_size = config.getPreparationBatchSize()
    if batch_size and batch_size > 0:
        return batch_size
    return 32


def create_many_attention_map_files_from_config(config, split_name):
    """
        Reads the image files from the sub-directories given as split names.
        
        Creates the according feature map files in the top directory.
    """
    target_shape = config.getImageInputShape()
    image_feature_size = config.getImageFeaturesSize()
    
    image_infix = get_infix_from_config(config, split_name)
    image_prefix = "COCO_" + image_infix
    
    bounding_boxes = load_prepared_boxes_json_from_config(config, split_name)
    
    boxes = calculate_metrics(bounding_boxes)
    
    boxes_by_id = collections.defaultdict(list)
    [boxes_by_id[box["image_id"]].append(box) for box in boxes]
    
    images_top_directory = config.getDatasetImagesDirectoryPath()
    image_paths = get_image_paths(to_split_dir(images_top_directory, split_name))
    
    processables = to_processables(image_paths, boxes_by_id, target_shape, image_prefix, image_feature_size)
    preprocess_bounding_boxes(processables)


def preprocess_bounding_boxes(processables, processes=20):
    split_dict_listing = __load_and_preprocess_data_into_parallel(processables, number_of_processes=processes)  
    for imaget in split_dict_listing:
        if imaget[0] == "Failure":
            print(imaget)  

    def collect_success(dicts):
        return [imaget[1] for imaget in dicts if imaget[0] == "Success"]

    split_dicts = collect_success(split_dict_listing)  
    return split_dicts


def to_processables(file_paths, boxes_by_id, target_shape, image_prefix, image_feature_size):
    dicts = []
    skipped = 0
    for file_path in file_paths:
        image_id = extract_image_id(file_path)
        d = {}
        d["path"] = file_path
        d["target_shape"] = (target_shape[0], target_shape[1])  # width, height
        d["boxes"] = boxes_by_id[image_id]
        d["image_id"] = image_id
        d["image_prefix"] = image_prefix
        d["image_feature_size"] = image_feature_size
        if len(d["boxes"]) == 0:  # defaultdict returns empty list
            #print("Warning: No bounding boxes for image id", image_id)
            skipped = skipped + 1
        else:
            dicts.append(d)
    if skipped > 0:
        print("Warning: Could not prepared bounding boxes for a total amount of images:", skipped)
    return dicts


def __load_and_preprocess_data_into_parallel(dicts, number_of_processes):
    results = []
    with Pool(processes=number_of_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(__load_and_preprocess_single_defended, dicts), total=len(dicts)):
            results.append(result)
    return results

        
def __load_and_preprocess_single_defended(imaged):
    try:
        load_and_preprocess_single(imaged)
        return ("Success", imaged)
    except:
        err_msg = sys.exc_info()[0]
        err = sys.exc_info()[1]
        error = (imaged["path"], err_msg, err)
        return ("Failure", error)


def load_and_preprocess_single(processable):
    file_path = processable["path"]
    target_shape = processable["target_shape"] 
    feature_size = processable["image_feature_size"]
    feature_shape = np.array([np.sqrt(feature_size), np.sqrt(feature_size)])
    feature_shape = feature_shape.astype("uint8")
    
    image_id = processable["image_id"]
    
    with load_img(file_path) as image:
        attention_maps = []
        bounding_boxes = processable["boxes"]
        for bounding_box in bounding_boxes:
            with Image.new(mode="L", size=(image.size)) as bbx:
                draw = ImageDraw.Draw(bbx)
                draw_box_on_image(draw, bounding_box["box"], fill=True)
                #bbx = bbx.resize(target_shape)
                bbx = bbx.resize(feature_shape, resample=NEAREST)
                attention_map = np.array(bbx)
                attention_map = np.reshape(attention_map, feature_size)
                attention_maps.append(attention_map)
    attention_maps = np.array(attention_maps)
    
    attention_ids = [bounding_box["box_id"] for bounding_box in processable["boxes"]]
    attention_ids = np.array(attention_ids)
    
    attention_labels = [bounding_box["category_id"]   for bounding_box in processable["boxes"]]
    attention_labels = np.array(attention_labels)
    
    directory_path = os.path.dirname(file_path)
    image_prefix = processable["image_prefix"]
    
    attention_map_path = to_image_path_by_id(image_prefix, image_id, directory_path, file_ending="bbx")
    store_numpy_to(attention_maps, attention_map_path)
    
    attention_label_path = to_image_path_by_id(image_prefix, image_id, directory_path, file_ending="lbx")
    store_numpy_to(attention_labels, attention_label_path)

    attention_id_path = to_image_path_by_id(image_prefix, image_id, directory_path, file_ending="ibx")
    store_numpy_to(attention_ids, attention_id_path)
    