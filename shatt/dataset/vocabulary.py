'''
Created on 14.03.2019

@author: Philipp
'''
import json

from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from shatt.dataset import load_json_from, store_json_to, determine_file_path
from shatt.dataset.tokenize import nltk_tokenize, tokenize_captions, keras_tokenize

DEFAULT_VOCABULARY_FILE_NAME = "mscoco_vocabulary.json"


def store_vocabulary(vocabulary, target_directory_path_or_file, split_name):
    return store_tokenizer(vocabulary.tokenizer, target_directory_path_or_file, split_name)


def store_tokenizer(tokenizer, target_directory_path_or_file, split_name):
    lookup_filename = DEFAULT_VOCABULARY_FILE_NAME
    if split_name:    
        lookup_filename = "mscoco_vocabulary_{}.json".format(split_name) 
    return store_json_to(json.loads(tokenizer.to_json()), target_directory_path_or_file, lookup_filename)


def get_vocabulary_file_path(config, split_name=None, flat=True):
    lookup_filename = DEFAULT_VOCABULARY_FILE_NAME
    if split_name and not flat:
        raise Exception("Only flat vocabulary path supported for now")
    if split_name and flat:
        lookup_filename = "mscoco_vocabulary_{}.json".format(split_name)
        # print("No support for split specific vocabulary loading. Please just name the file to use to " + lookup_filename)
    try:
        return determine_file_path(config.getDatasetTextDirectoryPath(), lookup_filename, to_read=True)
    except Exception:
        print("No vocabulary file found with name " + lookup_filename)
        return None


def create_vocabulary_file_from_config(config, captions, split_name, topmost_words=None, do_tokenize=True):
    """ Create vocabulary based on the prepared captions file """
    directory_path = config.getDatasetTextDirectoryPath()
    use_nltk = config.getPreparationUsingNltkTokenizer()
    return create_vocabulary_file(captions, directory_path, split_name, use_nltk, topmost_words, do_tokenize)

    
def create_vocabulary_file(captions, directory_path, split_name, use_nltk=False, vocabulary_size=None, do_tokenize=True):
    """
        Create the vocabulary based on the split.
        Expects the answers to be in the top directory.
        Puts the vocabulary in the top directory.
    """
    if captions and not isinstance(captions, list):
        raise Exception("Captions must be a listing of caption dicts")
    
    if do_tokenize:
        captions = tokenize_captions(captions, use_nltk)
        
    textual_captions = [["<S>"] + caption["tokenized"] + ["<E>"] for caption in captions]  # adding virtual <S> and <E> token
    
    """ we may use the vocabulary size as the top most allowed words """
    """ this is useful to create an auxiliary vocabulary to filter captions which would result in unknown words """
    """ nevertheless, the tokenizer will train on all words and use the word count when translating """
    tokenizer = Tokenizer(num_words=vocabulary_size, oov_token="UNK")
    print("Fit vocabulary on {} captions".format(len(captions)))
    tokenizer.fit_on_texts(textual_captions)
    
    return store_tokenizer(tokenizer, directory_path, split_name)


def load_vocabulary_file_from(directory_path_or_file, split_name=None, flat=True):
    """
        @param split_name: when given looks for the sub-directory or file in the flat directory
        @param flat: when True looks for a file in the given directory, otherwise looks into the sub-directory 
    """
    lookup_filename = DEFAULT_VOCABULARY_FILE_NAME
    
    if split_name and not flat:
        directory_path_or_file = "/".join([directory_path_or_file, split_name])
        
    if split_name and flat:
        lookup_filename = "mscoco_vocabulary_{}.json".format(split_name) 
        # print("No support for split specific vocabulary loading. Please just name the file to use to " + lookup_filename)
        
    tokenizer_config = load_json_from(directory_path_or_file, lookup_filename)
    tokenizer = tokenizer_from_json(json.dumps(tokenizer_config))
    return tokenizer


class Vocabulary():
    
    def __init__(self, tokenizer, use_nltk):
        self.tokenizer = tokenizer
        self.use_nltk = use_nltk
        
    def encodings_to_captions(self, encodings):
        return self.tokenizer.sequences_to_texts(encodings)
    
    def tokens_to_encoding(self, tokens):
        if tokens and not isinstance(tokens, list):
            raise Exception("Captions must be a listing of token listings")
            
        return self.tokenizer.texts_to_sequences(tokens)
    
    def get_size(self, include_padding=False):
        if include_padding:
            return len(self) + 1
        return len(self)
    
    def get_word_count(self, w):
        if w in self.tokenizer.word_counts:
            return self.tokenizer.word_counts[w]
        return 0
    
    def get_encoded_word_sequence(self, include_padding=False):
        """ this is actually something like range(vocabulary_size) """
        if include_padding:
            return [e for e in range(0, len(self) + 1)]
        return [e for e in range(1, len(self) + 1)]
        
    def get_word_sequence(self, padding_symbol=None):
        word_by_index = sorted(self.tokenizer.word_index.items(), key=lambda x: x[1])
        word_by_index = [w for w, _ in word_by_index]
        if padding_symbol:
            word_by_index.insert(0, padding_symbol)
        return word_by_index
    
    def __get_symbol(self, symbol):
        if symbol in self.tokenizer.word_index:
            return self.tokenizer.word_index[symbol]
        else:
            raise Exception("Cannot find start symbol {} in vocabulary".format(symbol))
    
    def get_end_token(self):
        return "<e>"
    
    def get_start_token(self):
        return "<s>"
    
    def get_start_symbol(self):
        return self.__get_symbol(self.get_start_token())
    
    def get_end_symbol(self):
        return self.__get_symbol(self.get_end_token())

    def __tokenize(self, caption, append_end_symbol):
        if isinstance(caption, dict):
            if "tokenized" in caption:
                caption = caption["tokenized"]
                if append_end_symbol:
                    if caption[-1] != self.get_end_token():
                        caption.append(self.get_end_token())
                return caption                    
            caption = caption["caption"]
            
        if self.use_nltk:
            caption = nltk_tokenize(caption)
        else:
            caption = keras_tokenize(caption)
            
        if append_end_symbol:
            if caption[-1] != self.get_end_token():
                caption.append(self.get_end_token())
        return caption

    def captions_to_encoding(self, captions, append_end_symbol=True, do_tokenize=True):
        """
            Encode the textual captions to an integer encoding. Unknown words are replaced with UNK.
            The integer max value is the vocabulary size.
            
            @param captions: the textual captions as list 
            @return: the captions as integer sequences
        """
        if captions and not isinstance(captions, list):
            raise Exception("Captions must be a listing of captions")
            
        if do_tokenize:
            captions = [self.__tokenize(c, append_end_symbol) for c in captions]
        
        return self.tokenizer.texts_to_sequences(captions)
    
    def __len__(self):
        return len(self.tokenizer.word_index)
    
    @staticmethod
    def create_vocabulary_from_config(config, split_name=None):
        return Vocabulary.create_vocabulary_from_vocabulary_json(config.getDatasetTextDirectoryPath(),
                                                                 split_name,
                                                                 config.getPreparationUsingNltkTokenizer(),
                                                                 )

    @staticmethod
    def create_vocabulary_from_vocabulary_json(source_directory_path_or_file, split_name, use_nltk):
        return Vocabulary(load_vocabulary_file_from(source_directory_path_or_file, split_name), use_nltk)


class PaddingVocabulary(Vocabulary):
    """
        The padding vocabulary additional zero pads the captions to maximal length when performing an encoding.
    """
    
    def __init__(self, tokenizer, use_nltk, captions_max_length):
        """
            @param captions_max_length: the (globally) maximal length of a caption
        """
        super().__init__(tokenizer, use_nltk)
        self.captions_max_length = captions_max_length

    def captions_to_encoding(self, captions, append_end_symbol=True):
        """
            Encode the textual captions to an integer encoding. Unknown words are replaced with UNK.
            The integer max value is the vocabulary size.
            
            The padding is attached at the end of each caption up to the maximal caption length.
            
            @param captions: the textual captions as list 
            @return: the padded and encoded captions
        """
        encoded_captions = super().captions_to_encoding(captions, append_end_symbol)
        padded_captions = pad_sequences(encoded_captions, maxlen=self.captions_max_length, padding="post")
        return padded_captions
    
    @staticmethod
    def create_vocabulary_from_config(config, split_name=None):
        return PaddingVocabulary.create_vocabulary_from_vocabulary_json(config.getDatasetTextDirectoryPath(),
                                                                        config.getPreparationUsingNltkTokenizer(),
                                                                        config.getCaptionMaximalLength(),
                                                                        split_name)
    
    @staticmethod
    def create_vocabulary_from_vocabulary_json(source_directory_path_or_file, use_nltk, captions_max_length, split_name=None):
        return PaddingVocabulary(load_vocabulary_file_from(source_directory_path_or_file, split_name), use_nltk, captions_max_length)
