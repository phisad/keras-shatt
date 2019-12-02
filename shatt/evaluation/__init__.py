

def split_box_captions(captions, split_category="0"):
    fix_captions = []
    bbx_captions = []
    for caption in captions:
        caption_cat = caption["category"]
        if caption_cat == split_category:
            fix_captions.append(caption)
        else:
            bbx_captions.append(caption)
    return fix_captions, bbx_captions


def split_box_captions_to_text(captions, split_category="0"):
    fix_captions, bbx_captions = split_box_captions(captions, split_category)
    fix_captions = [caption["caption"].split(" ") for caption in fix_captions]
    bbx_captions = [caption["caption"].split(" ") for caption in bbx_captions]
    return fix_captions, bbx_captions


import os


def to_model_path(model_dir, epoch, require_dir=False):
    if os.path.isdir(model_dir):
        model_name = "shatt.{:03}.h5".format(epoch)
        return model_dir + "/" + model_name
    else:
        if require_dir:
            raise Exception("Path must be a model directory, but is " + model_dir)
        else:
            return model_dir


def to_model_dir(path_to_model, require_path=False):
    if os.path.isdir(path_to_model):
        if require_path:
            raise Exception("Path must be a model checkpoint, but is " + path_to_model)
        else:
            return path_to_model
    model_dir = os.path.dirname(path_to_model)
    return model_dir
