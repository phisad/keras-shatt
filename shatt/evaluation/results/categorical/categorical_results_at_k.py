'''
Created on 23.05.2019

@author: Philipp
'''
from shatt.evaluation import split_box_captions
from collections import defaultdict
from jiwer import wer
import numpy as np
from shatt.evaluation.results import prepare_categories_at_k,\
    get_category_words_at_k, has_category_neighborhood, has_category_words_at_k
import logging


def create_categories_with_neighbors_at_k(categories_neighbors, k_list):
    categories_with_neighbors_by_id = defaultdict(list)
    for category in categories_neighbors:
        for cat_id in category["ids"]:
            categories_with_neighbors_by_id[cat_id].append(category)
    
    """ prepare neighbors at k"""
    prepare_categories_at_k(categories_with_neighbors_by_id, k_list)
    return categories_with_neighbors_by_id

def compute_filtered_categorical_matches_at_k(categories, categories_neighbors, box_captions, alternating_captions, do_strict=False, k_list=[1, 5, 10]):
    
    """ [{"category":str,"ids":[],"neighbors":[]}] """
    """ Warning: Its likely that some ids are referenced more than once e.g. for two-word categories"""
    categories_with_neighbors_by_id = create_categories_with_neighbors_at_k(categories_neighbors, k_list)
    
    """ [{"caption","category","image_id"}] """
    control_captions, box_captions = split_box_captions(box_captions)
    control_captions = dict([(c["image_id"], c) for c in control_captions])
    log = logging.getLogger()
    log.info("{0} {1}".format("total control (number of images)", len(control_captions)))
    
    """ add result attributes """
    for k in k_list:
        for box_caption in box_captions:
            box_caption["result@" + str(k)] = "unknown"  # ignore, correct, incorrect
            box_caption["result_relevant@" + str(k)] = True
            box_caption["result_unrelevant_hardly@" + str(k)] = False
            box_caption["result_unrelevant_softly@" + str(k)] = False
    
    """ remove boxes that match the control captions with fixed attention on the whole image (probably box unrelated captions) """
    """ [deactivate] for now: we only want to rely on the alternating caption """
    """ not checking for the control box caption seems to produce the same quantitative results !!! """
    """
    filter_relevant_categorical_matches(categories_with_neighbors_by_id, control_captions, box_captions,
                                                                      do_strict, k_list, run_name="check fixed")
    """

    """ remove boxes that match the alternating att (probably box unrelated captions, only new captions) """
    alternating_captions = dict([(c["image_id"], c) for c in alternating_captions])
    filter_relevant_categorical_matches(categories_with_neighbors_by_id, alternating_captions, box_captions,
                                                                      do_strict, k_list, run_name="check alternating")
    
    """ compute the matches """
    compute_categorical_matches(box_captions, categories_with_neighbors_by_id, do_strict, k_list)
    
    """ above the caption dicts themselves are tagged """
    """ therefore we can use them here to get all results """
    
    results = []
    processed_count = 0
    total_captions = len(box_captions)
    for c in box_captions:
        processed_count += 1
        print(">> Prepare results {:d}/{:d} ({:3.0f}%)".format(processed_count, total_captions, processed_count / total_captions * 100), end="\r")
        
        caption = c["caption"]
        image_id = c["image_id"]
        cat_id = c["category"]
        category = categories[int(cat_id)]["name"]
        control = control_captions[image_id]["caption"]
        if image_id not in alternating_captions: 
            selfatt = ""  # there is a missing batch
            c["result"] = "ignore"
        else:
            selfatt = alternating_captions[image_id]["caption"]
        
        result = {
                    "caption_box": caption,
                    "caption_control": control,
                    "caption_control_wer" : np.round(wer(control, caption), 2),
                    "caption_selfatt": selfatt,
                    "caption_selfatt_wer" : np.round(wer(selfatt, caption), 2),
                    "image_id": image_id,
                    "box_id" : c["box_id"],
                    "category": category,
                    "category_id": cat_id,
                    "result_neighborhood": [word for words in c["result_neighborhood@" + str(max(k_list))] for word in words]
                }
        for k in k_list:
            result["result@" + str(k)] = c["result@" + str(k)]
            result["result_relevant@" + str(k)] = c["result_relevant@" + str(k)]
            result["result_unrelevant_hardly@" + str(k)] = c["result_unrelevant_hardly@" + str(k)]
            result["result_unrelevant_softly@" + str(k)] = c["result_unrelevant_softly@" + str(k)]
        results.append(result)
    return results


def percentage(count, total_count, return_inverse=False):
    if return_inverse:
        return "{} %".format(100 - np.round(count / total_count * 100, 2))
    return "{} %".format(np.round(count / total_count * 100, 2))


def compute_categorical_matches(box_captions, categories_with_neighbors_by_id, do_strict=False, k_list=[1]):
    """
        When a category is contained in a caption, then it is a correct match, otherwise not.
    """
    log = logging.getLogger()
    for k in k_list:
        result_k = "result@" + str(k)
        result_neighborhood_k = "result_neighborhood@" + str(k)
        correct = []
        incorrect = []
        #ignore = []
        for box_caption in box_captions:
            category_words_neighborhood = get_category_words_at_k(box_caption, categories_with_neighbors_by_id, k)
            box_caption[result_neighborhood_k] = category_words_neighborhood
            """ ignore relevancy here for now """
            """ we keep track of the relevance anyway """
            """
            if not box_caption["result_relevant@" + str(k)]:
                ignore.append(box_caption)
                continue
            """
            if has_category_neighborhood(box_caption, category_words_neighborhood, None, do_strict):
                # count boxes that match the category (box relevant captions that are correct)
                box_caption[result_k] = "correct"
                correct.append(box_caption)
            else:
                # count boxes that dont match the category (box relevant captions that are wrong)
                box_caption[result_k] = "incorrect"
                incorrect.append(box_caption)
                
        total_box_count = len(box_captions)
        # the image ids may overlap when there are correct boxes and incorrect boxes on the same image
        log.info("{0} {1} {2}".format("-"*20, result_k, "-"*20))
        #log.info("{0} {1} {2}".format(result_k, "ignore", len(ignore)))
        #log.info("{0} {1} {2}".format(result_k, "ignore images", len(set([l["image_id"] for l in ignore]))))
        incorrect_count = len(incorrect)
        log.info("{0} {1} {2} -> {3} (total {4})".format(result_k, 
                                                         "incorrect", 
                                                         incorrect_count, 
                                                         percentage(incorrect_count, total_box_count), 
                                                         total_box_count))
        
        correct_count = len(correct)
        log.info("{0} {1} {2} -> {3} (total {4})".format(result_k, 
                                                         "correct  ", 
                                                         correct_count, 
                                                         percentage(correct_count, total_box_count), 
                                                         total_box_count))
        
        total_image_count = len(set([b["image_id"] for b in box_captions]))
        
        incorrect_image_count = len(set([l["image_id"] for l in incorrect]))
        log.info("{0} {1} {2} -> {3} (total {4})".format(result_k, 
                                                         "incorrect images", 
                                                         incorrect_image_count, 
                                                         percentage(incorrect_image_count, total_image_count), 
                                                         total_box_count))
        
        correct_image_count = len(set([l["image_id"] for l in correct]))
        log.info("{0} {1} {2} -> {3} (total {4})".format(result_k, 
                                                         "correct   images", 
                                                         correct_image_count, 
                                                         percentage(correct_image_count, total_image_count), 
                                                         total_box_count))

def filter_relevant_categorical_matches(categories_with_neighbors_by_id, check_captions, box_captions, do_strict=False, k_list=[1], run_name="filter relevant"):
    """
        ignore box captions that 
        a) [deactivated] are equal to the check captions (hard)
        b) [deactivated] that have categories contained in the check caption (soft) 
        c) an exception is thrown
        d) there is an alternating caption missing (see above)
    """
    # print(run_name, "total check captions", len(check_captions))
    # print(run_name, "total boxes", len(box_captions))
    log = logging.getLogger()
    log.info("{0} {1} {2}".format("-"*20, "relevancy", "-"*20))
    for k in k_list:
        error_count = 0
        # type a) that are exact matches (hardly unrelated, do not tell anything new about the image)
        hardly_unrelevant_captions = []
        # type b) that contain already the box category (softly unrelated, cannot tell anything new about the image)
        softly_unrelevant_captions = []
        
        for box_caption in box_captions:
            image_id = box_caption["image_id"]
            
            try:
                check_caption = check_captions[image_id]
                """ check if the same sentence """
                if box_caption["caption"] == check_caption["caption"]:
                    hardly_unrelevant_captions.append(box_caption)
                    box_caption["result_unrelevant_hardly@" + str(k)] = True  # actually doesnt depend on k
                    box_caption["result_relevant@" + str(k)] = False
                    """ we dont mark ignore here, but keep track of relevance """
                    """
                    box_caption["result@" + str(k)] = "ignore"
                    """
                    
                """ check if sentence contains already category words"""
                """ only filter for exact category occurrence """
                if has_category_words_at_k(box_caption, categories_with_neighbors_by_id, check_caption, do_strict, k=k):
                    box_caption["result_unrelevant_softly@" + str(k)] = True
                    box_caption["result_relevant@" + str(k)] = False
                    """ we dont mark ignore here, but keep track of relevance """
                    """
                    box_caption["result@" + str(k)] = "ignore"
                    """
                    softly_unrelevant_captions.append(box_caption)
                    
            except Exception:
                error_count = error_count + 1
                box_caption["result@" + str(k)] = "ignore"
                box_caption["result_relevant@" + str(k)] = False
        
        total_box_count = len(box_captions)
        total_box_count = total_box_count - error_count
        #log.info("{0} {1} {2}".format(run_name + "@" + str(k), "errors", error_count))
        #log.info("{0} {1} {2}".format(run_name + "@" + str(k), "hardly unrelevant", len(hardly_unrelevant_captions)))
        softly_unrelevant_count = len(softly_unrelevant_captions)
        log.info("{0} {1} {2} -> {3} {4} (new) ({5} errors)".format(run_name + "@" + str(k), 
                                          "softly unrelevant", 
                                          softly_unrelevant_count, 
                                          percentage(softly_unrelevant_count, total_box_count),
                                          percentage(softly_unrelevant_count, total_box_count, return_inverse=True),
                                          error_count))
