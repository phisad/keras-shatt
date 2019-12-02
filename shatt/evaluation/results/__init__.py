from shatt.dataset.boxes import load_categories_by_id
from shatt.dataset import load_json_from, store_json_to


def collect_capable_category_names(categories_with_neighbors, categorical_capability):
    missing_categories = []
    for c in categories_with_neighbors:
        cat_name = c["category"]
        for cc in categorical_capability:
            cap_name = cc["category"]["name"]
            if cat_name in cap_name.split(" "):
                cap_count = cc["count_" + cat_name]
                if not cap_count:
                    missing_categories.append(cat_name)
    capable_categories = [c for c in categories_with_neighbors if c["category"] not in missing_categories]
    return capable_categories


def has_category_words_at_k(box_caption, categories_with_neighbors_by_id, other_caption=None, do_strict=False, k=1):
    category_words_neighborhood = get_category_words_at_k(box_caption, categories_with_neighbors_by_id, k)
    return has_category_neighborhood(box_caption, category_words_neighborhood, other_caption, do_strict)

    
def has_category_neighborhood(box_caption, category_words_neighborhood, other_caption=None, do_strict=False):
    for category_words in category_words_neighborhood:
        caption = box_caption
        if other_caption:
            caption = other_caption
        if has_category_words(caption, category_words, do_strict):
            return True
    return False

    
def has_category_words(caption, category_words, do_strict=False):
    caption_text = caption["caption"].split(" ")
    if len(category_words) > 1 and do_strict:
        """ both words must be in the caption (but we cannot reproduce sequence here anymore) """
        return set(category_words).issubset(caption_text)
    for category_word in category_words:
        if category_word in caption_text:
            return True
    return False 


def get_category_words_at_k(box_caption, categories_with_neighbors_by_id, k):
    """
        This is necessary, because categories_with_neighbors_by_id may also exists of two words
        having a list of category words with neighbors instead of a single word with neighbors.
        
        Then we check for both of them.
    """
    cat_id = box_caption["category"]
    cat_id = int(cat_id)
    # when there are several category words, then both have the same neighborhood entry
    return categories_with_neighbors_by_id[cat_id][0]["neighborhood"][k]


def prepare_categories_at_k(categories_with_neighbors_by_id, k_list):
    from more_itertools import unique_everseen
    for categories_with_neighbors in categories_with_neighbors_by_id.values():
        category_neighborhood = []  # a list of word lists (b.c. of bi-gram categories)
        """ first add the category itself """
        if len(categories_with_neighbors) > 1:
            """ a bi-gram category """
            category_neighborhood.append([category["category"] for category in categories_with_neighbors])
            # zip neighborhoods and remove duplicates
            zipped_neighborhood_0 = [w for (w, _) in categories_with_neighbors[0]["neighbors"][1:]]
            zipped_neighborhood_1 = [w for (w, _) in categories_with_neighbors[1]["neighbors"][1:]]
            # max 2 words here
            zipped_neighborhood = []
            for idx in range(len(zipped_neighborhood_0)):
                zipped_neighborhood.append(zipped_neighborhood_0[idx])
                zipped_neighborhood.append(zipped_neighborhood_1[idx])
            zipped_neighborhood = unique_everseen(zipped_neighborhood)
            zipped_neighborhood = [[w] for w in zipped_neighborhood]
            category_neighborhood.extend(zipped_neighborhood)
        else:
            category = categories_with_neighbors[0]
            category_neighborhood.append([category["category"]])
            category_neighborhood.extend([[w] for (w, _) in category["neighbors"][1:]])  # ignore first entry
        for k in k_list:
            for category in categories_with_neighbors:  # usually only one entry
                if "neighborhood" not in category:
                    category["neighborhood"] = {}
                category["neighborhood"][k] = category_neighborhood[:k]


def is_correct_at_k(result, k, k_list, strict):
    result_pattern = "result@{}"
    is_correct = result[result_pattern.format(k)] == "correct"
    if not strict or k == 1:
        return is_correct
    if not is_correct:
        return False
    for other_k in k_list:
        if k != other_k and result[result_pattern.format(other_k)] == "correct":
            return False
    return True

