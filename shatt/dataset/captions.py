'''
Created on 03.04.2019

@author: Philipp
'''
from shatt import SPLIT_TRAINVAL, SPLIT_TRAIN, SPLIT_VALIDATE
from shatt.dataset import store_json_to, load_json_from, determine_file_path
from shatt.dataset.tokenize import tokenize_captions
from shatt.dataset.vocabulary import create_vocabulary_file_from_config, \
    load_vocabulary_file_from, Vocabulary
from collections import Counter

DEFAULT_PREPARED_CAPTIONS_FILE_NAME = "mscoco_captions.json"

DEFAULT_PREPARED_CAPTIONS_SPLIT_FILE_NAME_PATTERN = "mscoco_captions_{}.json"

DEFAULT_CAPTION_FILE_NAME = "captions.json"

import collections


def get_prepared_captions_file_path(config, split_name=None, flat=True):
    lookup_filename = DEFAULT_PREPARED_CAPTIONS_FILE_NAME
    if split_name and not flat:
        raise Exception("Only flat vocabulary path supported for now")
    if split_name and flat:
        lookup_filename = DEFAULT_PREPARED_CAPTIONS_SPLIT_FILE_NAME_PATTERN.format(split_name)
        # print("No support for split specific vocabulary loading. Please just name the file to use to " + lookup_filename)
    try:
        return determine_file_path(config.getDatasetTextDirectoryPath(), lookup_filename, to_read=True)
    except Exception:
        print("No vocabulary file found with name " + lookup_filename)
        return None


def to_caption_listing_by_image_id(captions):
    results = collections.defaultdict(list)
    for entry in captions:
        image_id = entry["image_id"]
        caption = entry["caption"]
        results[int(image_id)].append(caption)
    return results


def load_captions_json_from(directory_path_or_file, split_name=None, flat=False):
    """
        @param split_name: when given looks for the sub-directory or file in the flat directory
        @param flat: when True looks for a file in the given directory, otherwise looks into the sub-directory 
    """
    lookup_filename = DEFAULT_CAPTION_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        raise Exception("Not supported to have source caption files on the same level as the top dataset directory")
        
    return load_json_from(directory_path_or_file, lookup_filename)


def load_prepared_captions_json_from_config(config, split_name):
    return load_prepared_captions_json_from(config.getDatasetTextDirectoryPath(), split_name)


def load_prepared_captions_json_from(directory_path_or_file, split_name=None, flat=True):
    lookup_filename = DEFAULT_PREPARED_CAPTIONS_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:    
        lookup_filename = DEFAULT_PREPARED_CAPTIONS_SPLIT_FILE_NAME_PATTERN.format(split_name) 
        
    return load_json_from(directory_path_or_file, lookup_filename)

    
def store_prepared_captions_as_file(prepared_captions, target_directory_path_or_file, split_name=None):
    lookup_filename = DEFAULT_PREPARED_CAPTIONS_FILE_NAME
    if split_name:    
        lookup_filename = DEFAULT_PREPARED_CAPTIONS_SPLIT_FILE_NAME_PATTERN.format(split_name) 
    return store_json_to(prepared_captions, target_directory_path_or_file, lookup_filename)


def create_prepared_captions_file_from_config(config, split_names, topmost_words=None):
    if split_names and not isinstance(split_names, list):
        split_names = [split_names]
    return [__create_prepared_captions_file_from_config(config, split_name, topmost_words) for split_name in split_names]

"""
    Same as in the derived implementation of the original paper
"""


def preprocess_caption(caption):
    caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
    caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
    caption = " ".join(caption.split())  # replace multiple spaces
    return caption


def preprocess_captions(captions):
    for c in captions:
        c["caption"] = preprocess_caption(c["caption"])


def __create_prepared_captions_file_from_config(config, split_name, topmost_words=None):
    """ 
        For the normal train split we only filter the training captions
        to simulate other captions in the validate split
        for train+val split we have to filter both splits.
        
        In both cases we create a single captions.json file
    """
    directory_path = config.getDatasetTextDirectoryPath()
    if split_name == SPLIT_TRAINVAL:
        training_captions_json = load_captions_json_from(directory_path, SPLIT_TRAIN)
        valdiation_captions_json = load_captions_json_from(directory_path, SPLIT_VALIDATE)
        captions = training_captions_json["annotations"] + valdiation_captions_json["annotations"]
    else:
        # use training labels also for the validate split to filter the captions
        captions_json = load_captions_json_from(directory_path, split_name)
        captions = captions_json["annotations"]
        """
            A caption is an entry like
            {
                "image_id": 318556,
                "id": 48,
                "caption": "A very clean and well decorated empty bathroom"
            }
        """
    captions = captions
    print("Create prepared captions file")
    caption_maximal_length = config.getCaptionMaximalLength()
    total_amount = len(captions)
    
    print("Preprocess captions")
    preprocess_captions(captions)
    
    print("Tokenize captions")
    captions = tokenize_captions(captions, config.getPreparationUsingNltkTokenizer())
    
    """ the max length is supposed to include the end tag, but we dont have it here yet """
    """ plus one for the end symbol that is automatically added during training """
    """ the start symbol is provided by the caption model and therefore must not be prepended """
    print("Filter captions based on max length {:}".format(caption_maximal_length)) 
    prepared_captions = [caption for caption in captions if len(caption["tokenized"]) + 1 <= caption_maximal_length]
    reduced_amount = len(prepared_captions)
    print("Reduced captions from {} to {} ({:.2f}%)".format(total_amount, reduced_amount, reduced_amount / total_amount * 100))
    
    if topmost_words and topmost_words > 0:
        print("Filter captions based on max vocabulary size {:}".format(topmost_words))
        total_amount = len(prepared_captions)
        create_vocabulary_file_from_config(config, prepared_captions, "aux", topmost_words, do_tokenize=False)  # dont tokenize again
        aux_vocabulary = Vocabulary.create_vocabulary_from_config(config, "aux")
        
        tokenized_captions = [c["tokenized"] for c in prepared_captions] 
        caption_ids = [c["id"] for c in prepared_captions]
        
        def __encode_captions(vocabulary, tokenized_captions):
            total = len(tokenized_captions)
            processed_count = 0
            results = []
            for tokenized_caption in tokenized_captions:
                processed_count += 1
                print(">> Encoding caption {:d}/{:d} ({:3.0f}%)".format(processed_count, total, processed_count / total * 100), end="\r")
                encoded_caption = vocabulary.captions_to_encoding([tokenized_caption], append_end_symbol=False, do_tokenize=False)
                results.append(encoded_caption[0])
            return results
            
        encoded_captions = __encode_captions(aux_vocabulary, tokenized_captions)
        __write_unknown_words_file(encoded_captions, aux_vocabulary, directory_path, split_name)
        
        def __filter_captions(prepared_captions, encoded_captions, caption_ids):
            caption_to_id = zip(encoded_captions, caption_ids)
            
            """ check encodings for unknown words and keep ids when caption has no unknowns """
            total = len(encoded_captions)
            processed_count = 0
            filter_caption_ids = set()
            for caption, _id in caption_to_id:
                processed_count += 1
                print(">> Check caption {:d}/{:d} ({:3.0f}%)".format(processed_count, total, processed_count / total * 100), end="\r")
                if 1 not in caption:
                    filter_caption_ids.add(_id)
            
            """ filter the prepared captions to exclude the ones with unknown words """
            total = len(prepared_captions)
            processed_count = 0
            filtered_captions = []
            for p in prepared_captions:
                processed_count += 1
                print(">> Filter caption {:d}/{:d} ({:3.0f}%)".format(processed_count, total, processed_count / total * 100), end="\r")
                if p["id"] in filter_caption_ids:
                    filtered_captions.append(p)
            return filtered_captions

        prepared_captions = __filter_captions(prepared_captions, encoded_captions, caption_ids)
        reduced_amount = len(prepared_captions)
        print("Reduced captions from {} to {} ({:.2f}%)".format(total_amount, reduced_amount, reduced_amount / total_amount * 100))
    
    return store_prepared_captions_as_file(prepared_captions, directory_path, split_name)


def __write_unknown_words_file(encoded_captions, vocabulary, directory_path, split_name):
    counter = Counter([w for s in encoded_captions for w in s])
    
    json_content = []
    
    total_count = 0
    for s in encoded_captions:
        if 1 in s:
            total_count = total_count + 1
    json_content.append({"captions with unknown": total_count})
    
    caption_count = {}        
    for s in encoded_captions:
        count = sum([1 for w in s if w == 1])
        if count not in caption_count:
            caption_count[count] = 0
        caption_count[count] = caption_count[count] + 1
    keys = sorted(caption_count.keys())
    for k in keys:
        json_content.append({"unknown per caption " + str(k): caption_count[k]})
    
    for idx, total_count in counter.most_common():
        content = {"word" : vocabulary.tokenizer.index_word[idx], "total_count" : total_count}
        json_content.append(content)
    
    json_content.append(content)
    store_json_to(json_content, directory_or_file=directory_path, lookup_filename="mscoco_caption_counts_unknown_{}.json".format(split_name))
