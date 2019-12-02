
from shatt.dataset import load_json_from, store_json_to
from shatt.dataset.boxes import load_categories_by_id
from shatt.evaluation.results import is_correct_at_k
from shatt.evaluation.results.categorical.categorical_results_at_k import compute_filtered_categorical_matches_at_k
from shatt.experiments import RESULT_FILE_PATTERN
from shatt.evaluation.results.categorical.categorical_results_at_k_allowed import compute_filtered_categorical_matches_at_k_allowed


def compute_categorical_matches_from_model_dir_at_k(dataset_dir, model_dir, epoch, experiment_name="attention_fixed", do_strict=False, do_tight=True, k_list=[1]):
    """
        The computation will make use of the categories that are actually involved.
        Therefore all missing categories are ignored.
        
        Furthermore the computation will provide scores for k-nearest categorical words.
        Therefore the word embeddings and nearest neighbors must be computed before.
        
        In addition, the neighbors file provides a flatten view on the categories 
        (bigrams mapped to unigrams) to easier compute the matches via word matching in the caption.
    """
    import logging
    import sys
    logging.basicConfig(level=logging.DEBUG,
                        format="%(message)s",
                        handlers=[
                            logging.FileHandler("{0}/{1}.log".format(model_dir, "categorical_results")),
                            logging.StreamHandler(sys.stdout)
                        ])
    
    """ the categories also with two-word occurrences to produce the output"""
    categories = load_categories_by_id(dataset_dir, "validate")
    
    """ the flatten categories with neighbors """ 
    categories_neighbors = load_json_from(model_dir, lookup_filename="category_neighbors.json")
    
    """ the experimental results for each box and each image """
    box_results_file_name = RESULT_FILE_PATTERN.format("{}_epoch_{:03}".format(experiment_name, epoch))
    box_captions = load_json_from(model_dir, box_results_file_name)
    
    """ the result captions per image on epoch end """
    results_file_name = RESULT_FILE_PATTERN.format("validate_epoch_{:03}".format(epoch))
    alternating_captions = load_json_from(model_dir, results_file_name)
    
    if do_tight:
        results = compute_filtered_categorical_matches_at_k(categories, categories_neighbors, box_captions, alternating_captions, do_strict, k_list)
    #else: Never invoked for thesis results
        #results = compute_filtered_categorical_matches_at_k_allowed(categories, categories_neighbors, box_captions, alternating_captions, do_strict, k_list)
    store_json_to(results, model_dir, lookup_filename="categorical_results_all.json")
    # strict meaning that only those that full fill only k or more are stored
    for k in k_list:
        store_json_to([r for r in results if is_correct_at_k(r, k, k_list, strict=True)], model_dir, lookup_filename="categorical_results_correct_at_{}.json".format(k))
