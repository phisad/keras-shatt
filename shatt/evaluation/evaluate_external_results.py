'''
Created on 10.08.2019

@author: Philipp
'''
import nltk
from nltk.corpus import stopwords
from shatt.dataset import store_json_to, load_json_from
from shatt.dataset.tokenize import tokenize_nltk_single


def evaluate_file(results_dir, results_file, control_file, selfatt_file, ignore_missing=False):
    """
        [{
            "caption": "hot dogs and buns on a hot dog bun",
            "question_info": {
                "answer": "red",
                "answer_type": "other",
                "image_id": 42,
                "question": "What color is the flip flop?",
                "question_id": 421
            },
            "type": "word"
        },]
        
        for each entry look if
            caption 
                stemmed words (excluding stop words)
            are contained in 
                stemmed words (excluding stop words)
                of 
                    answer
                    question
                
        (keep track of question infos for control and self-att)
    """ 
    result_listing, skip_list = enrich_result_listing(results_dir, results_file, control_file, selfatt_file)
    
    if not ignore_missing:
        skip_list = []
    result_listing = evaluate_results(result_listing, skip_list)
    
    file_name = "extern_results.json"
    if ignore_missing:
        file_name = "extern_results_ignore_missing.json"
    store_json_to(result_listing, results_dir, file_name)


def enrich_result_listing(results_dir, results_file, control_file, selfatt_file):
    """
        for each question add a caption of
            control
            self-attention
    """
    result_listing = load_json_from(results_dir, results_file)
    
    """ get control captions """ 
    control_captions = load_json_from(results_dir, control_file)
    control_by_image_id = dict([(int(c["image_id"]), c) for c in control_captions])
    
    """ get control captions """ 
    selfatt_captions = load_json_from(results_dir, selfatt_file)
    selfatt_by_image_id = dict([(int(c["image_id"]), c) for c in selfatt_captions])
    
    question_by_id = dict([(r["question_info"]["question_id"], r["question_info"]) for r in result_listing])
    other_captions = []
    skipped_control = []
    skipped_selfatt = []
    
    processed_count = 0
    expected_num_batches = len(question_by_id.values())
    for question_info in question_by_id.values():
        processed_count = processed_count + 1
        if processed_count % 10000 == 0:
            print(">> Enriching question {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
        
        image_id = int(question_info["image_id"])
        if image_id in control_by_image_id and image_id in selfatt_by_image_id:
            other_captions.append({
                "caption" : control_by_image_id[image_id]["caption"],
                "question_info" : question_info,
                "type" : "control"
                })
        else:
            skipped_control.append(image_id) 
            continue
        
        if image_id in control_by_image_id and image_id in selfatt_by_image_id:
            other_captions.append({
                "caption" : selfatt_by_image_id[image_id]["caption"],
                "question_info" : question_info,
                "type" : "selfatt"
                })
        else:
            skipped_selfatt.append(image_id)
            continue
        
    if len(skipped_control) > 0:
        print("Skipped control", len(skipped_control), skipped_control)
        
    if len(skipped_selfatt) > 0:
        print("Skipped selfatt", len(skipped_selfatt), skipped_selfatt)
        
    return result_listing + other_captions, skipped_control + skipped_selfatt


def evaluate_results(result_listing, skip_list=[]):
    stop_words = set(stopwords.words('english'))
    stemmer = nltk.PorterStemmer()
    
    """ could be an ignored set, when we ignore skip list """
    evaluated_result_listing = []
    ignore_list = []
    processed_count = 0
    expected_num_batches = len(result_listing)
    for result in result_listing:
        processed_count = processed_count + 1
        if processed_count % 20000 == 0:
            print(">> Processing captions {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
        
        image_id = result["question_info"]["image_id"]
        if image_id in skip_list:
            ignore_list.append(image_id)
            continue
        
        tokenized_caption = tokenize_nltk_single(result["caption"])
        """ TODO use keras tokenize for question and answer as in experiment """
        tokenized_question = tokenize_nltk_single(result["question_info"]["question"])
        tokenized_answer = tokenize_nltk_single(result["question_info"]["answer"])
        
        no_stop_caption = set([w for w in tokenized_caption if w not in stop_words])
        no_stop_question = set([w for w in tokenized_question if w not in stop_words])
        
        stemmed_caption = set([stemmer.stem(w) for w in no_stop_caption])
        stemmed_question = set([stemmer.stem(w) for w in no_stop_question])
        answer = set([stemmer.stem(w) for w in tokenized_answer])
        
        result["caption_in_question"] = list(stemmed_caption & stemmed_question)
        result["caption_in_answer"] = list(stemmed_caption & answer)
        
        """ could be an ignored set, when we ignore skip list """
        evaluated_result_listing.append(result)
        
    if len(ignore_list) > 0:
        print("ignored", len(ignore_list))
        
    return evaluated_result_listing
