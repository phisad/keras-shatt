'''
Created on 10.08.2019

@author: Philipp
'''
from shatt.dataset import store_json_to, load_json_from

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
sns.set(color_codes=True)

def analyse_file(results_dir, results_file, target_file="extern_results_statistics.json"):
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
            "type": "word",
            "caption_in_question" : [],
            "caption_in_answer" : []
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
    result_listing = load_json_from(results_dir, results_file)
    statistics = analyse_listing(result_listing)
    store_json_to(statistics, results_dir, target_file)


def analyse_listing(result_listing):
    """
        for each type: [word, phrase, question, control, selfatt]
            count + total_count
                if caption in question
                if caption in answer 
                if caption in question or answer
    """
    statistics = {
        "word" : {"caption_in_question":0, "caption_in_answer":0, "caption_in_question_or_answer":0,
                  "caption_in_question_total":0, "caption_in_answer_total":0, "caption_in_question_or_answer_total":0},
        "phrase" : {"caption_in_question":0, "caption_in_answer":0, "caption_in_question_or_answer":0,
                    "caption_in_question_total":0, "caption_in_answer_total":0, "caption_in_question_or_answer_total":0},
        "question" : {"caption_in_question":0, "caption_in_answer":0, "caption_in_question_or_answer":0,
                      "caption_in_question_total":0, "caption_in_answer_total":0, "caption_in_question_or_answer_total":0},
        "control" : {"caption_in_question":0, "caption_in_answer":0, "caption_in_question_or_answer":0,
                     "caption_in_question_total":0, "caption_in_answer_total":0, "caption_in_question_or_answer_total":0},
        "selfatt" : {"caption_in_question":0, "caption_in_answer":0, "caption_in_question_or_answer":0,
                     "caption_in_question_total":0, "caption_in_answer_total":0, "caption_in_question_or_answer_total":0}
        }
    processed_count = 0
    expected_num_batches = len(result_listing)
    for result in result_listing:
        processed_count = processed_count + 1
        print(">> Processing captions {:d}/{:d} ({:3.0f}%)".format(processed_count, expected_num_batches, processed_count / expected_num_batches * 100), end="\r")
        caption_in_question = result["caption_in_question"]
        caption_in_answer = result["caption_in_answer"]
        
        stats = statistics[ result["type"]]
        caption_in_question_count = len(caption_in_question)
        caption_in_answer = len(caption_in_answer)
        
        if caption_in_question_count > 0:
            stats["caption_in_question"] = stats["caption_in_question"] + 1
        
        if caption_in_answer > 0: 
            stats["caption_in_answer"] = stats["caption_in_answer"] + 1
            
        if caption_in_question_count > 0 or caption_in_answer > 0:
            stats["caption_in_question_or_answer"] = stats["caption_in_question_or_answer"] + 1
            
        stats["caption_in_question_total"] = stats["caption_in_question_total"] + caption_in_question_count
        stats["caption_in_answer_total"] = stats["caption_in_answer_total"] + caption_in_answer
        stats["caption_in_question_or_answer_total"] = stats["caption_in_question_or_answer_total"] + caption_in_question_count + caption_in_answer
        
    return statistics


def barplot_analyse_file(results_dir, results_file, relative_count=None):
    """
        @param relative_count: the total counts
            if given, then the results are given in percent
    """
    """
    {
        "control": {
            "caption_in_answer": 7566,
            "caption_in_answer_total": 7961,
            "caption_in_question": 13071,
            "caption_in_question_or_answer": 18175,
            "caption_in_question_or_answer_total": 22342,
            "caption_in_question_total": 14381
        },
        "phrase": {
            "caption_in_answer": 8122,
            "caption_in_answer_total": 8527,
            "caption_in_question": 13274,
            "caption_in_question_or_answer": 18883,
            "caption_in_question_or_answer_total": 23144,
            "caption_in_question_total": 14617
        },
        "question": {
            "caption_in_answer": 8813,
            "caption_in_answer_total": 9332,
            "caption_in_question": 13840,
            "caption_in_question_or_answer": 19823,
            "caption_in_question_or_answer_total": 24661,
            "caption_in_question_total": 15329
        },
        "selfatt": {
            "caption_in_answer": 9160,
            "caption_in_answer_total": 9646,
            "caption_in_question": 15904,
            "caption_in_question_or_answer": 21563,
            "caption_in_question_or_answer_total": 27456,
            "caption_in_question_total": 17810
        },
        "word": {
            "caption_in_answer": 11269,
            "caption_in_answer_total": 11889,
            "caption_in_question": 15495,
            "caption_in_question_or_answer": 23125,
            "caption_in_question_or_answer_total": 29098,
            "caption_in_question_total": 17209
        }
    }
    """
     
    statistics = load_json_from(results_dir, results_file)
    """
        dataframe = {
                     "type":["word","phrase","question","control","selfatt"], 
                     "summary":["total","distinct"], 
                     "in":["answer","question","both"]
                     }
    """
    dataframe = []
    for type in statistics:
        count_once = "One Word Match per Caption "
        count_all = "All Word Matches per Caption"
        stats = statistics[type]
        if relative_count:
            y_axis_label = "Relative Counts in Percent ({} Questions in Total)".format(relative_count)
            dataframe.append({y_axis_label : stats["caption_in_answer"] / relative_count * 100, "Attention Type":type , "Counting": count_once, "Words":"in Answer"})
            dataframe.append({y_axis_label : stats["caption_in_answer_total"] / relative_count * 100, "Attention Type":type , "Counting":count_all, "Words":"in Answer"})
            dataframe.append({y_axis_label : stats["caption_in_question"] / relative_count * 100, "Attention Type":type , "Counting":count_once, "Words":"in Question"})
            dataframe.append({y_axis_label : stats["caption_in_question_total"] / relative_count * 100, "Attention Type":type , "Counting":count_all, "Words":"in Question"})
            dataframe.append({y_axis_label : stats["caption_in_question_or_answer"] / relative_count * 100, "Attention Type":type , "Counting":count_once, "Words":"in Either-Or"})
            dataframe.append({y_axis_label : stats["caption_in_question_or_answer_total"] / relative_count * 100, "Attention Type":type , "Counting":count_all, "Words":"in Either-Or"})
        else:
            y_axis_label = "Counts"
            dataframe.append({y_axis_label : stats["caption_in_answer"], "Attention Type":type , "Counting": count_once, "Words":"in Answer"})
            dataframe.append({y_axis_label : stats["caption_in_answer_total"], "Attention Type":type , "Counting":count_all, "Words":"in Answer"})
            dataframe.append({y_axis_label : stats["caption_in_question"], "Attention Type":type , "Counting":count_once, "Words":"in Question"})
            dataframe.append({y_axis_label : stats["caption_in_question_total"], "Attention Type":type , "Counting":count_all, "Words":"in Question"})
            dataframe.append({y_axis_label : stats["caption_in_question_or_answer"], "Attention Type":type , "Counting":count_once, "Words":"in Either-Or"})
            dataframe.append({y_axis_label : stats["caption_in_question_or_answer_total"], "Attention Type":type , "Counting":count_all, "Words":"in Either-Or"})
    
    df = pd.DataFrame(dataframe) 
    #df["type"] = df["type"].astype("category")
    #sns.catplot(data=df, x="Attention Type", y="Counts", hue="summary", col="in", kind="bar", ci=None, aspect=.6)
    sns.catplot(data=df, x="Attention Type", y=y_axis_label, hue="Words", col="Counting", kind="bar", 
                order=["word","phrase","question","control","selfatt"], orient="v", legend_out=False)
    #plt.legend(loc="inside")
    plt.show()