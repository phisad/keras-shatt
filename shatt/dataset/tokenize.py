'''
Created on 26.04.2019

@author: Philipp
'''
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
import tqdm
import sys


def tokenize_captions(captions, use_nltk):
    """
        A caption is an entry like
        {
            "image_id": 318556,
            "id": 48,
            "caption": "A very clean and well decorated empty bathroom"
        }
    """
    if use_nltk:
        captions = __tokenize_nltk_parallel(captions, 12)
    else:
        captions = __tokenize_keras_parallel(captions, 12)
    return captions


def keras_tokenize(caption):
    return text_to_word_sequence(caption.lower())


def __tokenize_keras_parallel(dicts, number_of_processes):
    print("Tokenize using Keras with {:d} processes".format(number_of_processes))
    results = []
    with Pool(processes=number_of_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(__tokenize_keras_single_defended, dicts), total=len(dicts)):
            state = result[0]
            caption = result[1]
            if state == "Success":
                results.append(caption)
            else:
                print(caption)
    return results

        
def __tokenize_keras_single_defended(captiond):
    try:
        __tokenize_keras_single(captiond)
        return ("Success", captiond)
    except:
        err_msg = sys.exc_info()[0]
        err = sys.exc_info()[1]
        error = (captiond["caption"], err_msg, err)
        return ("Failure", error)


def __tokenize_keras_single(captiond):
    captiond["tokenized"] = replace_or_remove_numbers(keras_tokenize(captiond["caption"]))


def nltk_tokenize(caption):
    return word_tokenize(str(caption).lower())


def __tokenize_nltk_parallel(dicts, number_of_processes):
    """
        A caption is an entry like
        {
            "image_id": 318556,
            "id": 48,
            "caption": "A very clean and well decorated empty bathroom"
        }
    """
    print("Tokenize using NLTK with {:d} processes".format(number_of_processes))
    results = []
    with Pool(processes=number_of_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(__tokenize_nltk_single_defended, dicts), total=len(dicts)):
            state = result[0]
            caption = result[1]
            if state == "Success":
                results.append(caption)
            else:
                print(caption)
    return results

        
def __tokenize_nltk_single_defended(captiond):
    try:
        __tokenize_nltk_single(captiond)
        return ("Success", captiond)
    except:
        err_msg = sys.exc_info()[0]
        err = sys.exc_info()[1]
        error = (captiond["caption"], err_msg, err)
        return ("Failure", error)


import re


def replace_or_remove_numbers(prepared_caption):
    result = []
    for t in prepared_caption:
        if t == "1":
            t = "one"
        elif t == "2":
            t = "two"
        elif t == "3":
            t = "three"
        elif t == "4":
            t = "four"
        elif t == "5":
            t = "five"
        elif t == "6":
            t = "six"
        elif t == "7":
            t = "seven"
        elif t == "8":
            t = "eight"
        elif t == "9":
            t = "nine"
        elif t == "0":
            t = "zero"
        if re.match(r'\d+', t):
            pass
        else:
            result.append(t)
    return result


def __tokenize_nltk_single(captiond):
    filtered_caption = text_to_word_sequence(captiond["caption"])
    tokenized_caption = nltk_tokenize(" ".join(filtered_caption))
    tokenized_caption = replace_or_remove_numbers(tokenized_caption)
    captiond["tokenized"] = tokenized_caption 


def tokenize_nltk_single(caption):
    filtered_caption = text_to_word_sequence(caption)
    tokenized_caption = nltk_tokenize(" ".join(filtered_caption))
    tokenized_caption = replace_or_remove_numbers(tokenized_caption)
    return tokenized_caption 