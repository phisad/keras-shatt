#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser

from shatt import SPLIT_VALIDATE, SPLIT_TRAIN
from shatt.configuration import Configuration
from shatt.dataset.vocabulary import Vocabulary, get_vocabulary_file_path, create_vocabulary_file_from_config
from shatt.dataset.captions import create_prepared_captions_file_from_config, load_prepared_captions_json_from_config, store_prepared_captions_as_file, \
    get_prepared_captions_file_path, load_captions_json_from
from shatt.dataset import captions
from shatt.dataset.boxes import create_prepared_boxes_file,\
    load_prepared_boxes_json_from, calculate_metrics


def __get_split_or_default(run_opts, default_split):
    split_name = default_split
    if run_opts.split_name:
        split_name = run_opts.split_name
    return split_name


def __get_splits_or_default(run_opts, default_splits):
    split_names = default_splits
    if run_opts.split_name:
        split_names = [run_opts.split_name]
    return split_names


def main():
    parser = ArgumentParser("Prepare the MSCOCO dataset for caption generator training")
    parser.add_argument("command", help="""One of [captions, vocabulary, encode, boxes, all]. 
                        1.Step| captions: Fetch all captions from the annotations file.
                        2.Step| vocabulary: Create the vocabulary. Also determines vocabulary size and captions maximal length. Requires to create the caption file before. 
                        3.Step| encode: Add the vocabulary based encoding to the caption entries. Requires to create the vocabulary file before.
                        (Opt) | boxes: Read the instances file and write a boxes json without segmentation infos. Required to generate box image files.
                        all: All of the above""")
    parser.add_argument("-t", "--topmost_words", type=int, help="Use the configured vocabulary size as a topmost amount of words for the vocabulary.")
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-s", "--split_name", help="""The split name to prepare the training data for. One of [train, validate, test, trainval]. 
                        The split name determines sub-directory to lookup in the dataset directories.
                        A special split name is 'trainval' which combines both train and validate splits.
                        Notice: When the split name is not specified for training then the standard split of train and validate will be prepared. 
                        This is usually the wanted behavior. It is not possible for now to combine e.g. train and test.
                        """)
    
    run_opts = parser.parse_args()
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("Starting preparation: {}".format(run_opts.command))
    
    if run_opts.command in ["karpathy"]:
        print("\nAlt. Step: Transform the karpathy split file to our prepare captions file format")
        split_name = "karpathy"
        directory_path = config.getDatasetTextDirectoryPath()
        karpathy_captions_json = load_captions_json_from(directory_path, split_name)
        """
            {
             "images":
                [
                    {"filepath","sentids","filename","imgid","split",
                    "sentences":[{"tokens":[],"raw","sentid"}],"cocoid"}
                ]
        """
        prepared_captions = []
        """
            [{"caption","encoded","id","image_id","tokenized"}]
        """
        image_counter = 0
        for entry in karpathy_captions_json["images"]:
            if entry["filepath"] == "val2014":
                image_counter = image_counter + 1
                for sentence in entry["sentences"]:
                    prepared_caption = {}
                    prepared_caption["caption"] = " ".join(sentence["tokens"])
                    prepared_caption["tokenized"] = sentence["tokens"]
                    prepared_caption["id"] = sentence["sentid"]
                    prepared_caption["image_id"] = entry["cocoid"]
                    prepared_captions.append(prepared_caption)
        print("Prepared images", image_counter)
        print("Prepared captions", len(prepared_captions))
        store_prepared_captions_as_file(prepared_captions, directory_path, split_name)
        
    if run_opts.command in ["all", "captions"]:
        print("\n1. Step: Filter the original captions file to those that fulfill max caption length (to reduce generator complexity)")
        split_names = __get_splits_or_default(run_opts, [SPLIT_TRAIN, SPLIT_VALIDATE])
        prepared_captions_path = get_prepared_captions_file_path(config, "train")
        if prepared_captions_path:
            print("Found prepared captions file at " + prepared_captions_path)
        else:
            print("Creating prepared caption file for " + str(split_names))
            """ we already filter the captions here, when we want to reduce the vocabulary size """
            """ then later the actual vocabulary is built that will know all prepared caption words """
            """ 
                we have to get the upper bound for the vocabulary here b.c. it might be still less than the configured value
                for example when there are words that only occur in captions that have unknown words after the filtering
                then the vocabulary size will be less than the configured value, so that it would have to be adjusted again
                therefore we take the upper bound here from an script argument 
            """
            create_prepared_captions_file_from_config(config, split_names, topmost_words=run_opts.topmost_words)
        
    if run_opts.command in ["all", "vocabulary"]:
        print("\n2. Step: Create the vocabulary from the filtered captions")
        split_name = __get_split_or_default(run_opts, SPLIT_TRAIN)
        vocabulary_file_path = get_vocabulary_file_path(config, split_name) 
        if vocabulary_file_path:
            print("Found vocabulary file at " + vocabulary_file_path)
        else:
            print("Creating vocabulary file for split " + split_name)
            captions = load_prepared_captions_json_from_config(config, split_name)
            vocabulary_file_path = create_vocabulary_file_from_config(config, captions, split_name, do_tokenize=False)  # already prepared captions
        
        vocab = Vocabulary.create_vocabulary_from_vocabulary_json(vocabulary_file_path, split_name=None, use_nltk=config.getPreparationUsingNltkTokenizer())
        vocab_size = len(vocab) 
        print("Determined vocabulary size " + str(vocab_size))
        print("Determined vocabulary start symbol {}".format(vocab.get_start_symbol()))
        print("Determined vocabulary end symbol {}".format(vocab.get_end_symbol()))

        split_names = __get_splits_or_default(run_opts, [SPLIT_TRAIN, SPLIT_VALIDATE])
        for split_name in split_names:    
            captions = load_prepared_captions_json_from_config(config, split_name)
            maximal = max([len(caption["tokenized"]) for caption in captions]) + 1  # virual end tag
            print("Sanitary check: Determined maximal captions length in {} split: {} ".format(split_name, str(maximal)))

    if run_opts.command in ["all", "encode"]:
        print("\n3. Step: Encode the filtered captions using the vocabulary")
        split_name = __get_split_or_default(run_opts, SPLIT_TRAIN)
        vocabulary_file_path = get_vocabulary_file_path(config, split_name) 
        vocab = Vocabulary.create_vocabulary_from_vocabulary_json(vocabulary_file_path, split_name=None, use_nltk=config.getPreparationUsingNltkTokenizer())

        split_names = __get_splits_or_default(run_opts, [SPLIT_TRAIN, SPLIT_VALIDATE])
        for split_name in split_names:    
            captions = load_prepared_captions_json_from_config(config, split_name)
            total_count = len(captions)
            processed_count = 0
            for caption in captions:
                processed_count += 1
                print('>> Encode captions %d/%d' % (processed_count, total_count), end="\r")
                caption["encoded"] = vocab.tokens_to_encoding([caption["tokenized"]])[0]
            store_prepared_captions_as_file(captions, config.getDatasetTextDirectoryPath(), split_name)

    if run_opts.command in ["all", "boxes"]:
        print("\n(Optional): Read the instances file and write a boxes json without segmentation infos. Required to generate box image files.")
        split_name = __get_split_or_default(run_opts, SPLIT_TRAIN)
        file_path = create_prepared_boxes_file(config, split_name)
        boxes_json = load_prepared_boxes_json_from(file_path)
        calculate_metrics(boxes_json)
        

if __name__ == '__main__':
    main()
    
