'''
Created on 03.07.2019

    Accurate entries are those that are correct and new. Here we most likely ignore correct box captions that produced by boxes framing main objects in the image.
    
    {
        "box_id": "318568",
        "caption_box": "a striped bed consisting of a tall green bed table and table",
        "caption_control": "a man standing in a kitchen with a sink",
        "caption_control_wer": 1.11,
        "caption_selfatt": "a man is standing in a kitchen with a refrigerator",
        "caption_selfatt_wer": 1.0,
        "category": "bed",
        "category_id": "65",
        "image_id": "402639",
        "result@1": "correct",
        "result@5": "correct"
        "result_relevant@1": true,
        "result_relevant@5": true
    }
@author: Philipp
'''
from shatt.dataset import load_json_from
import numpy as np


def percentage(count, total_count, return_inverse=False):
    if return_inverse:
        return "{} %".format(100 - np.round(count / total_count * 100, 2))
    return "{} %".format(np.round(count / total_count * 100, 2))


def compute_accuracy(results_dir, k_list=[], total_count=None):
    """
        @param total_count: int
            When the total count is given, then also the percentage is computed.
    """
    file_name = "categorical_results_all.json"
    results = load_json_from(results_dir, lookup_filename=file_name)
    total_count = len(results)
    for k in k_list:
        accurate_count = 0
        relevant_count = 0
        for r in results:
            prop_name_correct = "result@{}".format(k)
            """ TODO something is wrong with the relevant flag """
            #prop_name_relevant = "result_relevant@{}".format(k)
            prop_name_relevant = "result_unrelevant_softly@{}".format(k)
            is_relevant = not r[prop_name_relevant]
            if is_relevant:
                relevant_count = relevant_count + 1
                if r[prop_name_correct] == "correct":
                    accurate_count = accurate_count + 1

        accurate_k = "accurate@{}".format(k)
        print("{0} {1} {2}".format("-"*20, accurate_k, "-"*20))
        print("{0} {1} {2}".format(accurate_k, "total", total_count))
        print("{0} {1} {2} {3}".format(accurate_k, "accurate", accurate_count, percentage(accurate_count, relevant_count)))
        if total_count:
            print("{0} {1} {2} {3}".format(accurate_k, "relevant", relevant_count, percentage(relevant_count, total_count)))
