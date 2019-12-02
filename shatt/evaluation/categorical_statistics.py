'''
Created on 29.05.2019

@author: Philipp
'''
from shatt.dataset import load_json_from
from shatt.dataset.boxes import load_categories_json_from, load_categories_by_id
from shatt.configuration import Configuration
import collections
from math import log

#model_dir = "F:/Development/shatt-models/v10k/alpha005/results_tight_not"
model_dir = "F:/Development/shatt-models/v10k/alpha005/tight"

"""
    {
        "box_id": "1097218",
        "caption_box": "a bathroom with a sink and toilet and a sink",
        "caption_control": "a bathroom with a sink and a mirror",
        "caption_control_wer": 0.38,
        "caption_selfatt": "a bathroom with a sink and a mirror",
        "caption_selfatt_wer": 0.38,
        "category": "toilet",
        "category_id": "70",
        "image_id": "380854",
        "result@1": "correct",
        "result@2": "correct",
        "result@3": "correct",
        "result@4": "correct",
        "result@5": "correct",
        "result_neighborhood": ["toilet", "toilette", "commode","refrigerator","sink"],
        "result_relevant": true,
        "result_unrelevant_hardly": false,
        "result_unrelevant_softly": false
    },
    
    {
        "id": 1,
        "name": "person",
        "supercategory": "person"
    }
"""
import pandas
import seaborn as sns
sns.set(color_codes=True)

import numpy as np
from matplotlib import pyplot as plt


def corrects_per_category():
    for k in [1,3,5]:
        corrects_per_category_at_k(k)
        

def compute_categorical_matches(categories, results, k, distinct=True):
    """
        @return: a listing of dict
            The dicts as {"category_id",
                          "count_relevant",
                          "count_match",
                          "count_nomatch",
                          "count_total",
                          "match_percent",
                          "total_percent",
                          "log_count_total" }
    """
    stats = []
    for r in results:
        s = {}
        s["category_id"] = int(r["category_id"])
        s["match"] = r["result@" + str(k)] == "correct"
        s["relevant"] = r["result_relevant@" + str(k)]
        #s["relevant"] = r["result_relevant"]
        stats.append(s)
    
    for c in categories.values():
        c["count_relevant"] = 0
        c["count_match"] = 0
        c["count_nomatch"] = 0
        c["count_total"] = 0
    
    for s in stats:
        c = categories[s["category_id"]]
        if distinct: # look if relevant
            if s["relevant"]:
                if s["match"]:
                    c["count_match"] = c["count_match"] + 1
                else:
                    c["count_nomatch"] = c["count_nomatch"] + 1
                c["count_relevant"] = c["count_relevant"] + 1
        elif s["match"]:
            c["count_match"] = c["count_match"] + 1 # otherwise just look if matches
        else:
            c["count_nomatch"] = c["count_nomatch"] + 1
        c["count_total"] = c["count_total"] + 1
        
    stats = sorted(categories.values(), key=lambda x:(x["count_total"], x["count_match"]), reverse=True)
    
    for s in stats:
        if distinct:
            s["count_total"] = s["count_relevant"]
        match_percent = np.round(s["count_match"] / s["count_total"], 2) if s["count_total"] > 0 else 0
        total_percent = np.round(s["count_total"] / len(results), 2)
        s["match_percent"] = match_percent
        s["total_percent"] = total_percent
        s["log_count_total"] = log(s["count_total"], 10)
        
    return stats

def corrects_per_category_at_k(model_dir, k, distinct=False, print_latex=True, do_plot=False, do_full=False):
    config = Configuration()
    categories = load_categories_by_id(config.getDatasetTextDirectoryPath(), split_name="validate")
    # results = load_json_from(model_dir, lookup_filename="categorical_results_correct_at_1.json")
    results = load_json_from(model_dir, lookup_filename="categorical_results_all.json")
    stats = compute_categorical_matches(categories, results, k, distinct)
    
    # df = pandas.DataFrame.from_dict(stats)
    # sns.jointplot(x="log_count_total", y="count_match", data=df)
    # g.ax_joint.set(xscale="log")
    
    # order histogram by matches
    # better not on a log scale ^^
    stats = sorted(stats, key=lambda x: x["match_percent"], reverse=True)
    total_sum = sum([s["count_total"] for s in stats])
    total_matches = sum([s["count_match"] for s in stats])
    print(total_sum)
    print(total_matches / total_sum)
    
    total_sum = sum([s["count_total"] for s in stats if s["name"] != "person"])
    total_matches = sum([s["count_match"] for s in stats if s["name"] != "person"])
    print(total_sum)
    print(total_matches / total_sum)
    
    for idx, s in enumerate(stats):
        if print_latex:
            if idx < 10 or idx > 69 or True:
                if do_full:
                    print("{:>2} & {:>2} & {:<15} & {:4d} & {:.0f} \\% & {:5d} & {:.0f} \\% \\\\".format(idx+1, s["id"], 
                                                                         s["name"], s["count_match"], s["match_percent"] * 100, 
                                                                         s["count_total"], s["total_percent"] * 100))
                else:                                                       
                    print("{:>2} & {:<15} & {:.0f} \\% & {:4d} \\\\".format(idx+1, s["name"], s["match_percent"] * 100, s["count_match"]))
                
            if idx == 10 or idx == 69:
                print("\\hline")
        else:
            print("{:<15} : {:4d} {:.2f}% / {:5d} {:.2f}%".format(s["name"], s["count_match"], s["match_percent"], s["count_total"], s["total_percent"]))
    
    if not do_plot:
        return
    
    bars = [s["name"] for s in stats]
    ind = np.arange(len(bars))
    def one_or_zero(idx):
        if idx % 2 == 0:
            return 0.5
        else:
            return 0
        
    #spaces = [one_or_zero(idx) for idx in range(len(ind))]
    #ind = ind + spaces
    width = 0.4
    
    
    no_matches = [s["count_nomatch"] for s in stats]
    matches = [s["count_match"] for s in stats]
    
    fig, ax1 = plt.subplots()
    percent_matches = [s["match_percent"] for s in stats]
    
    def shift(ind):
        #spaces = [one_or_zero(idx) for idx in range(len(ind))]
        ind = ind + 0.4
        return ind
    
    ax1.plot(ind, percent_matches, width, color="purple", linewidth=2, alpha=0.5)
    rect4 = ax1.bar(ind, percent_matches, 0.2, align='edge', color="purple", linewidth=0, alpha=0.5)
    if distinct:
        ax1.set_ylim([0,0.32])
        ax1.set_yticks(np.arange(0, .33, .02))
    
    ind_shifted = shift(ind) 
    ax2 = ax1.twinx()
    rect1 = ax2.bar(ind_shifted, [s["count_total"] for s in stats], width, log=True, align='edge', color="lightskyblue", linewidth=0)
    #rect4 = ax2.bar(ind, [s["count_relevant"] for s in stats], width, align='edge', color="navajowhite", linewidth=0)
    rect3 = ax2.bar(ind_shifted, no_matches, width, align='edge', color="salmon", linewidth=0, alpha=0.5)
    rect2 = ax2.bar(ind_shifted, matches, width, bottom=no_matches, align='edge', color="lightgreen", linewidth=0)
    ax2.set_ylabel('log-Counts')
    ax2.set_ylim([0,100000])
    
    if distinct:
        if k > 1:
            plot_title = "Box Captions including the Category Word or one of the {} Nearest Words (on the Distinct Subset)".format(k)
        else:
            plot_title = "Box Captions including the Category Word (on the Distinct Subset)"
    else:
        if k > 1:
            plot_title = "Box Captions including the Category Word or one of the {} Nearest Words ".format(k)
        else:
            plot_title = "Box Captions including the Category Word"
            
    ax2.set_title(plot_title)
    ax2.set_xticks(ind + width)
    ax2.set_xticklabels(bars)
    ax2.legend((rect1, rect2, rect4, rect3), ("Total Count", "Matching Count", "Matching Percent", "Not Matching"))
    
    
    
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
        tick.set_fontsize(8) 

    # plt.xticks(ind, bars)
    #fig.tight_layout()
    plt.show()

    
def main():
    stats = []
    for idx in range(10):
        name = "B"
        if idx > 5:
            name = "A"
        stats.append({"idx": idx, "match": idx % 2 == 0, "name":name})
    df = pandas.DataFrame.from_dict(stats)
    # Create bars
    
    # sns.jointplot(x="idx", y="idx", data=df)
    # sns.barplot(x="name", y="match", data=df)
    plt.show()

    
if __name__ == "__main__":
    main()
