'''
Created on 07.03.2019

@author: Philipp
'''
import configparser
import os.path as path
import os
import numpy as np

import json
from shatt.scripts import OPTION_DRY_RUN
SECTION_DATASET = "DATASETS"
OPTION_TEXTUAL_DATASET_DIRECTORY_PATH = "TextualDatasetDirectoryPath"
OPTION_TEXTUAL_PREPARATION_CORPUS = "TextualPreparationCorpus"
OPTION_VOCABULAY_INCLUDE_ANSWERS = "VocabularyIncludeAnswers"
OPTION_IMAGE_DATASET_DIRECTORY_PATH = "ImageDatasetDirectoryPath"
OPTION_TRAINING_IMAGE_NAME_INFIX = "TrainingImageNameInfix"
OPTION_NUMBER_OF_TRAINING_IMAGES = "NumberOfTrainingImages"
OPTION_VALIDATION_IMAGE_NAME_INFIX = "ValidationImageNameInfix"
OPTION_NUMBER_OF_VALIDATION_IMAGES = "NumberOfValidationImages"
OPTION_TEST_IMAGE_NAME_INFIX = "TestImageNameInfix"
OPTION_NUMBER_OF_TEST_IMAGES = "NumberOfTestImages"
OPTION_PREPARATION_BATCH_SIZE = "PreparationBatchSize"
OPTION_PREPARATION_USING_NLTK_TOKENIZER = "PreparationUsingNltkTokenizer"
OPTION_MEDIAN_BOX_WIDTH = "MedianBoxWidth"
OPTION_MEDIAN_BOX_HEIGHT = "MedianBoxHeight"

SECTION_MODEL = "MODEL"
OPTION_PRINT_MODEL_SUMMARY = "PrintModelSummary"
OPTION_MODEL_DERIVATE_NAME = "ModelDerivateName"
OPTION_USE_MAX_SAMPLER = "UseMaxSampler"
OPTION_CAPTION_MAXIMAL_LENGTH = "CaptionMaximalLength"
OPTION_VOCABULARY_SIZE = "VocabularySize"
OPTION_VOCABULARY_START_SYMBOL = "VocabularyStartSymbol"
OPTION_IMAGE_OUTPUT_LAYER = "ImageOutputLayer"
OPTION_IMAGE_FEATURES_SIZE = "ImageFeaturesSize"
OPTION_IMAGE_INPUT_SHAPE = "ImageInputShape"
OPTION_IMAGE_FEATURES_SIZE = "ImageFeaturesSize"
OPTION_BYPASS_IMAGE_FEATURES_COMPUTATION = "ByPassImageFeaturesComputation"
OPTION_IMAGE_TOP_LAYER = "ImageTopLayer"
OPTION_IMAGE_TOP_LAYER_DROPOUT_RATE = "ImageTopLayerDropoutRate"
OPTION_DROPOUT_RATE = "DropoutRate"
OPTION_ALPHA_REGULARIZATION = "AlphaRegularization"

SECTION_TRAINING = "TRAINING"
OPTION_GPU_DEVICES = "GpuDevices"
OPTION_TENSORBOARD_LOGGING_DIRECTORY = "TensorboardLoggingDirectory"
OPTION_EPOCHS = "Epochs"
OPTION_BATCH_SIZE = "BatchSize"
OPTION_MODEL_TYPE = "ModelType"
OPTION_USE_MULTI_PROCESSING = "UseMultiProcessing"
OPTION_WORKERS = "Workers"
OPTION_MAX_QUEUE_SIZE = "MaxQueueSize"

FILE_NAME = "configuration.ini"


def store_configuration(configuration, target_directory_path_or_file, split_name):
    lookup_filename = FILE_NAME
    if split_name:    
        lookup_filename = "configuration_{}.ini".format(split_name) 
    return store_config_to(configuration.config, target_directory_path_or_file, lookup_filename)


def store_config_to(config, directory_or_file, lookup_filename=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename, to_read=False)
    print("Persisting configuration to " + file_path)    
    with open(file_path, "w") as config_file:
        config.write(config_file)
    return file_path


def determine_file_path(directory_or_file, lookup_filename, to_read=True):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = directory_or_file
    if os.path.isdir(directory_or_file):
        if lookup_filename == None:
            raise Exception("Cannot determine source file in directory without lookup_filename")
        file_path = "/".join([directory_or_file, lookup_filename])
    if to_read and not os.path.isfile(file_path):
        raise Exception("There is no such file in the directory to read: " + file_path)
    return file_path


class Configuration(object):

    def __init__(self, config_path=None):
        '''
        Constructor
        '''
        self.run_opts = {}
        self.config = configparser.ConfigParser()
        if not config_path:
            config_path = Configuration.config_path()
        print("Use configuration file at: " + config_path)
        self.config.read(config_path)
        
    def __getitem__(self, idx):
        return self.run_opts[idx]
    
    def __setitem__(self, key, value):
        self.run_opts[key] = value

    def is_dryrun(self):
        if self[OPTION_DRY_RUN]:
            return self[OPTION_DRY_RUN]
        return False

    def getPrintModelSummary(self):
        return self.config.getboolean(SECTION_MODEL, OPTION_PRINT_MODEL_SUMMARY)
    
    def getModelDerivateName(self):
        return self.config.get(SECTION_MODEL, OPTION_MODEL_DERIVATE_NAME)
    
    def getUseMaxSampler(self):
        return self.config.getboolean(SECTION_MODEL, OPTION_USE_MAX_SAMPLER)
    
    def getCaptionMaximalLength(self):
        return self.config.getint(SECTION_MODEL, OPTION_CAPTION_MAXIMAL_LENGTH)
    
    def getVocabularySize(self):
        return self.config.getint(SECTION_MODEL, OPTION_VOCABULARY_SIZE)

    def getVocabularyStartSymbol(self):
        return self.config.getint(SECTION_MODEL, OPTION_VOCABULARY_START_SYMBOL)
    
    def getVocabularyIncludeAnswers(self):
        return self.config.getboolean(SECTION_DATASET, OPTION_VOCABULAY_INCLUDE_ANSWERS)
    
    def getPreparationCorpus(self):
        """
            One of [open-ended, multiple-choice]
        """
        return self.config.get(SECTION_DATASET, OPTION_TEXTUAL_PREPARATION_CORPUS)
    
    def getPreparationUsingNltkTokenizer(self):
        return self.config.getboolean(SECTION_DATASET, OPTION_PREPARATION_USING_NLTK_TOKENIZER)
    
    def getDatasetTextDirectoryPath(self):
        return self.config.get(SECTION_DATASET, OPTION_TEXTUAL_DATASET_DIRECTORY_PATH)
    
    def getDatasetImagesDirectoryPath(self):
        return self.config.get(SECTION_DATASET, OPTION_IMAGE_DATASET_DIRECTORY_PATH)
    
    def getTrainingImageNameInfix(self):
        return self.config.get(SECTION_DATASET, OPTION_TRAINING_IMAGE_NAME_INFIX)
    
    def getValidationImageNameInfix(self):
        return self.config.get(SECTION_DATASET, OPTION_VALIDATION_IMAGE_NAME_INFIX)
    
    def getTestImageNameInfix(self):
        return self.config.get(SECTION_DATASET, OPTION_TEST_IMAGE_NAME_INFIX)
    
    def getNumberOfTrainingImages(self):
        return self.config.getint(SECTION_DATASET, OPTION_NUMBER_OF_TRAINING_IMAGES)
    
    def getNumberOfValidationImages(self):
        return self.config.getint(SECTION_DATASET, OPTION_NUMBER_OF_VALIDATION_IMAGES)
    
    def getNumberOfTestImages(self):
        return self.config.getint(SECTION_DATASET, OPTION_NUMBER_OF_TEST_IMAGES)
    
    def getPreparationBatchSize(self):
        return self.config.getint(SECTION_DATASET, OPTION_PREPARATION_BATCH_SIZE)
    
    def getMedianBoxWidth(self):
        return self.config.getfloat(SECTION_DATASET, OPTION_MEDIAN_BOX_WIDTH)
    
    def getMedianBoxHeight(self):
        return self.config.getfloat(SECTION_DATASET, OPTION_MEDIAN_BOX_HEIGHT)
    
    def getImageInputShape(self):
        shape = self.config.get(SECTION_MODEL, OPTION_IMAGE_INPUT_SHAPE)
        shape_tuple = tuple(map(int, shape.strip('()').split(',')))
        return shape_tuple
    
    def getImageOutputLayer(self):
        return self.config.get(SECTION_MODEL, OPTION_IMAGE_OUTPUT_LAYER)
    
    def getImageFeaturesSize(self):
        return self.config.getint(SECTION_MODEL, OPTION_IMAGE_FEATURES_SIZE)
    
    def getImageTopLayer(self):
        return self.config.getboolean(SECTION_MODEL, OPTION_IMAGE_TOP_LAYER)
    
    def getImageTopLayerDropoutRate(self):
        return self.config.getfloat(SECTION_MODEL, OPTION_IMAGE_TOP_LAYER_DROPOUT_RATE)
    
    def getDropoutRate(self):
        return self.config.getfloat(SECTION_MODEL, OPTION_DROPOUT_RATE)
    
    def getAlphaRegularization(self):
        return self.config.getfloat(SECTION_MODEL, OPTION_ALPHA_REGULARIZATION)
    
    def getByPassImageFeaturesComputation(self):
        return self.config.getboolean(SECTION_MODEL, OPTION_BYPASS_IMAGE_FEATURES_COMPUTATION)
    
    def getGpuDevices(self):
        return self.config.getint(SECTION_TRAINING, OPTION_GPU_DEVICES)
    
    def getTensorboardLoggingDirectory(self):
        return self.config.get(SECTION_TRAINING, OPTION_TENSORBOARD_LOGGING_DIRECTORY)

    def getEpochs(self):
        return self.config.getint(SECTION_TRAINING, OPTION_EPOCHS)

    def getBatchSize(self):
        return self.config.getint(SECTION_TRAINING, OPTION_BATCH_SIZE)

    def getModelType(self):
        return self.config.get(SECTION_MODEL, OPTION_MODEL_TYPE)
    
    def getUseMultiProcessing(self):
        return self.config.getboolean(SECTION_TRAINING, OPTION_USE_MULTI_PROCESSING)    

    def getWorkers(self):
        return self.config.getint(SECTION_TRAINING, OPTION_WORKERS)    

    def getMaxQueueSize(self):
        return self.config.getint(SECTION_TRAINING, OPTION_MAX_QUEUE_SIZE)    
    
    def dump(self):
        print("Configuration:")
        for section in self.config.sections():
            print("[{}]".format(section))
            for key in self.config[section]:
                print("{} = {}".format(key, self.config[section][key]))
                
    @staticmethod
    def config_path():
        # Lookup file in project root or install root
        project_root = os.path.dirname(os.path.realpath(__file__))
        config_path = "/".join([project_root, FILE_NAME])
        if path.exists(config_path):
            return config_path
        print("Warn: No existing 'configuration.ini' at default location " + config_path)
        
        # Lookup file in user directory
        from pathlib import Path
        home_directory = str(Path.home())
        config_path = "/".join([home_directory, "shatt-" + FILE_NAME])
        if path.exists(config_path):
            return config_path
        print("Warn: No existing 'shatt-configuration.ini' file at user home " + config_path)
        
        raise Exception("""Please place a 'configuration.ini' in the default location 
                            or a 'shatt-configuration.ini' in your home directory 
                            or use the run option to specify a specific file""")

