'''
Created on 24.05.2019

@author: Philipp
'''
from collections import Counter
from shatt.dataset import load_json_from, store_json_to
from shatt.dataset.boxes import load_categories_json_from
from shatt.dataset.captions import load_prepared_captions_json_from
from shatt.evaluation import split_box_captions_to_text
from shatt.experiments import RESULT_FILE_PATTERN


def analyse_capability(captions, categories, name, model_dir):
    category_names = set([w for c in categories for w in c["name"].split(" ")])
    word_counter = Counter([w for c in captions for w in c if w in category_names])
    
    print(name, "Total", len(category_names))
    missing_cats = [c for c in category_names if c not in word_counter]
    print(name, "Existing", len(category_names) - len(missing_cats))
    print(name, "Missing", len(missing_cats), missing_cats)
    
    analysed_categories = []
    for c in categories:
        analysed_category = {}
        analysed_category["category"] = c
        for w in c["name"].split(" "):
            analysed_category["count_{}".format(w)] = word_counter[w]  # is missing when count = 0
        analysed_categories.append(analysed_category)
    store_json_to(analysed_categories, model_dir, lookup_filename="categorical_capability_{}.json".format(name))


def categorical_capability_from_config(config, model_dir, epoch, experiment_name="attention_fixed"):
    box_results_file_name = RESULT_FILE_PATTERN.format("{}_epoch_{:03}".format(experiment_name, epoch))
    results_file_name = RESULT_FILE_PATTERN.format("validate_epoch_{:03}".format(epoch))
    dataset_dir = config.getDatasetTextDirectoryPath()
    categorical_capability(dataset_dir, model_dir, box_results_file_name, results_file_name)

    
def categorical_capability(dataset_dir, model_dir, box_results_file_name, results_file_name):
    """
        How many categories are exposed in the captions, if so what is there count ?
    """
    """ [{"id","name","supercategory"}] """
    categories = load_categories_json_from(dataset_dir, "validate")
    
    """ all ground-truth captions (with tokenized word-list attribute) """
    gt_captions = load_prepared_captions_json_from(dataset_dir, "validate")
    gt_captions = [c["tokenized"] for c in gt_captions]
    analyse_capability(gt_captions, categories, "gt", model_dir)
    
    """ all captions with fixed attention over the whole image """
    box_captions = load_json_from(model_dir, box_results_file_name)
    fix_captions, bbx_captions = split_box_captions_to_text(box_captions)
    analyse_capability(fix_captions, categories, "fixed", model_dir)
    analyse_capability(bbx_captions, categories, "boxes", model_dir)
    
    """ all captions with alternating attention over the whole image """
    alt_captions = load_json_from(model_dir, results_file_name)
    alt_captions = [c["caption"].split(" ") for c in alt_captions]
    analyse_capability(alt_captions, categories, "alternating", model_dir)
