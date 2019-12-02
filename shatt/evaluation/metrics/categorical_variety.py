'''
Created on 03.07.2019

    How different is a box caption compared to the alternating caption using the word-error-rate?
    
    Whats the total variety?
    
    Whats the distribution of variety?
    
    Correlate newness or correctness with variety?
    
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
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)

def percentage(count, total_count, return_inverse=False):
    if return_inverse:
        return "{} %".format(100 - np.round(count / total_count * 100, 2))
    return "{} %".format(np.round(count / total_count * 100, 2))

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def compute_effected_variety(results_dir, target_caption, ylimit=None, plot=False):
    """ excluding WER scores = 0.0 """
    
    file_name = "categorical_results_all.json"
    results = load_json_from(results_dir, lookup_filename=file_name)
    print()
    wer_scores = np.array([r["caption_selfatt_wer"] for r in results 
                           if r["caption_box"] != r[target_caption]])
    total_scores = len(wer_scores)
    print("total scores", total_scores)
    
    wer_scores = reject_outliers(wer_scores)
    without_outliers = len(wer_scores)
    print("remaining scores", without_outliers)
    print("outliers", total_scores - without_outliers)
    
    wer_min = np.min(wer_scores)
    wer_max = np.max(wer_scores)
    wer_range = [wer_min, wer_max]
    print("wer range", wer_range)
    print("wer mean", np.round(np.mean(wer_scores), 2))
    print("wer median", np.median(wer_scores))
    
    #bins = np.abs(wer_min) + np.abs(wer_max)
    bins = np.arange(wer_min, wer_max, step=0.1)
    print("histogram bins", bins)
    hist, bin_edges = np.histogram(wer_scores, bins)
    print(hist)
    
    if plot:
        if ylimit:
            plt.ylim(0, ylimit)
        plt.hist(wer_scores, bins)
        #sns.distplot(wer_scores, bins)
        plt.show()

def compute_variety(results_dir, ylimit=None, plot=False):
    file_name = "categorical_results_all.json"
    results = load_json_from(results_dir, lookup_filename=file_name)
    print()
    wer_scores = np.array([r["caption_selfatt_wer"] for r in results])
    total_scores = len(wer_scores)
    print("total scores", total_scores)
    
    wer_scores = reject_outliers(wer_scores)
    without_outliers = len(wer_scores)
    print("remaining scores", without_outliers)
    print("outliers", total_scores - without_outliers)
    
    wer_min = np.min(wer_scores)
    wer_max = np.max(wer_scores)
    wer_range = [wer_min, wer_max]
    print("wer range", wer_range)
    print("wer mean", np.round(np.mean(wer_scores), 2))
    print("wer median", np.median(wer_scores))
    
    #bins = np.abs(wer_min) + np.abs(wer_max)
    bins = np.arange(wer_min, wer_max, step=0.1)
    print("histogram bins", bins)
    hist, bin_edges = np.histogram(wer_scores, bins)
    print(hist)
    
    if plot:
        if ylimit:
            plt.ylim(0, ylimit)
        plt.hist(wer_scores, bins)
        #sns.distplot(wer_scores, bins)
        plt.show()

def compute_indistinct(results_dir):
    file_name = "categorical_results_all.json"
    results = load_json_from(results_dir, lookup_filename=file_name)
    print()
    
    indistinct_control_captions = [r for r in results if r["caption_box"] == r["caption_control"]]
    print("indistinct control count", len(indistinct_control_captions), 
          percentage(len(indistinct_control_captions), len(results)),
          percentage(len(indistinct_control_captions), len(results), return_inverse=True)
          )
    
    indistinct_selfatt_captions = [r for r in results if r["caption_box"] == r["caption_selfatt"]]
    print("indistinct selfatt count", len(indistinct_selfatt_captions), 
          percentage(len(indistinct_selfatt_captions), len(results)),
          percentage(len(indistinct_selfatt_captions), len(results), return_inverse=True)
          )
    
    