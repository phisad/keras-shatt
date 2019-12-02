'''
Created on 24.05.2019

@author: Philipp
'''
import numpy as np
from shatt.dataset import load_json_from, store_json_to
from shatt.dataset.images import load_numpy_from, store_numpy_to
import collections
from shatt.dataset.vocabulary import Vocabulary
from tensorflow.python.keras.engine.input_layer import Input
from shatt.model import create_shatt_model_v2
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.models import Model
from shatt.evaluation import to_model_path
from shatt.dataset.boxes import load_textual_categories_by_id


class WordSequence():
    
    def __init__(self, encoded_word_sequence, batch_size):
        self.encoded_word_sequence = encoded_word_sequence
        self.batch_size = batch_size
    
    def get_num_batches(self):
        return int(np.ceil(len(self.encoded_word_sequence) / float(self.batch_size))) 
    
    def get_batch(self, idx):
        return self.encoded_word_sequence[idx * self.batch_size:(idx + 1) * self.batch_size]
    
    def one_shot_iterator(self):
        for idx in range(self.get_num_batches()):
            yield self.get_batch(idx)

            
def categorical_knearest_neighbors_from_config(config, model_dir, k=10):
    directory_path = config.getDatasetTextDirectoryPath()
    categorical_knearest_neighbors_from_category_dir(model_dir, directory_path, k)

    
def categorical_knearest_neighbors_from_category_dir(model_dir, category_dir, k=10):
    categories = load_textual_categories_by_id(category_dir, "validate")
    categorical_knearest_neighbors_from_model_dir(model_dir, categories, k)


def categorical_knearest_neighbors_from_model_dir(model_dir, categories, k=10):
    word_sequence = load_json_from(model_dir, "word_sequence.json")
    word_embeddings = load_numpy_from(model_dir, "word_embeddings.npy")
    categories_with_neighbors = categorical_knearest_neighbors(categories, word_sequence, word_embeddings, k)
    store_json_to(categories_with_neighbors, model_dir, "category_neighbors.json")

    
def categorical_knearest_neighbors(categories, word_sequence, word_embeddings, k=40):
    """
        Requires to compute word embeddings first.
    """
    
    """ there are two word categories """
    flatten_categories = set()
    for cat in categories.values():
        cat = cat.split(" ")
        if len(cat) > 1:
            for subcat in cat:
                flatten_categories.add(subcat)
        else:
            flatten_categories.add(cat[0])
    flatten_categories = list(flatten_categories)
    
    """ validate existence """
    for cat in flatten_categories:
        if cat not in word_sequence:
                print("Category not found", cat)
    
    """ fit nearest neighbors """            
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(word_embeddings)
    
    """ get nearest neighbors """
    category_indicies = [word_sequence.index(cat) for cat in flatten_categories]
    category_embeddings = word_embeddings[category_indicies]
    category_neighbor_distances, category_neighbor_indicies = knn.kneighbors(category_embeddings , n_neighbors=k, return_distance=True)
    
    """ combine results """
    categories_with_neighbors = []
    categories_by_name = dict([(name, idx) for idx, name in categories.items()])
    flatten_categories_by_name = collections.defaultdict(set)
    for flatten in flatten_categories:
        for category in categories_by_name:
            if flatten in category.split(" "):
                flatten_categories_by_name[flatten].add(categories_by_name[category])
                
    for idx, category in enumerate(flatten_categories):
        neighbor_words = [(word_sequence[widx], str(np.round(wist, 2))) for widx, wist in zip(category_neighbor_indicies[idx], category_neighbor_distances[idx])]
        category_ids = list(flatten_categories_by_name[category])
        category_with_neighbors = {"ids" : category_ids, "category":category, "neighbors": neighbor_words}
        categories_with_neighbors.append(category_with_neighbors)
    categories_with_neighbors = sorted(categories_with_neighbors, key=lambda x: sum(x["ids"]))   
    return categories_with_neighbors


def compute_word_embeddings(model_dir, epoch):
    path_to_model = to_model_path(model_dir, epoch)
    
    """ lookup model vocabulary """
    vocabulary = Vocabulary.create_vocabulary_from_vocabulary_json(model_dir, "", use_nltk=False)
    
    """ prepare and load model """
    vinput = Input((196, 512))
    model = create_shatt_model_v2(image_features_graph=(vinput, vinput),
                                  caption_max_length=16, vocabulary_size=len(vocabulary), dropout_rate=0.,
                                  start_encoding=vocabulary.get_start_symbol(), image_features_dimensions=196,
                                  embedding_size=512, hidden_size=1024, inference_mode=True,
                                  attention_graph=None, return_attention=True, use_max_sampler=True)
    model.load_weights(path_to_model, by_name=True)
    
    """ establish embedding model """
    layer_name = "shatt_word_embeddings"
    layer = model.get_layer(layer_name)
    if layer == None:
        raise Exception("Cannot find layer with name " + layer_name)
    input_words = Input(shape=(1,), name="embedding_callback_input_words")
    layer_output = layer(input_words)
    layer_output = Flatten(name="embedding_callback_flatten")(layer_output)
    embedding_model = Model(inputs=input_words, outputs=layer_output)
    
    """ write metadata.tsv """ 
    word_sequence = vocabulary.get_word_sequence(padding_symbol="<PAD>")
    store_json_to(word_sequence, model_dir, lookup_filename="word_sequence.json")
    
    """ encode sequence""" 
    encoded_word_sequence = vocabulary.get_encoded_word_sequence(include_padding=True)
    
    sequence = WordSequence(encoded_word_sequence, 64)
    
    processed_count = 0
    expected_num_batches = sequence.get_num_batches()
    results = []
    try:
        for words in sequence.one_shot_iterator():
            words = np.expand_dims(words, axis=-1)
            word_embeddings = embedding_model.predict_on_batch(words)
            results.extend(word_embeddings)
            processed_count = processed_count + 1
            print(">> Computing word embeddings {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
    except Exception as e:
        print("Exception: ", e)
    results = np.array(results)
    store_numpy_to(results, model_dir, lookup_file_name="word_embeddings.npy")
