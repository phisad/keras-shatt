'''
Created on 03.05.2019

@author: Philipp
'''
from shatt.dataset.images.providers import MscocoFileSystemImageProvider
from shatt.dataset.captions import to_caption_listing_by_image_id, \
    load_prepared_captions_json_from_config
from shatt.dataset.images import get_infix_from_config
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from PIL import ImageFilter
import numpy as np
from PIL.Image import LANCZOS, NEAREST, BOX


def show_many(images, max_rows, max_cols, figsize=(20, 20), titles=None, titles_hspace=.2, plot_title=None):
    plt.figure(figsize=figsize, dpi=300)

    for idx, image in enumerate(images):
        row = idx // max_cols
        col = idx % max_cols

        ax = plt.subplot2grid((max_rows, max_cols), (row, col))
        if titles != None:
            subtitle_conf = {"fontsize": 2}
            ax.set_title(titles[idx], fontdict=subtitle_conf, loc="left", pad=2)
        ax.axis("off")
        ax.imshow(image, aspect="auto")
        
    if titles == None:
        plt.subplots_adjust(wspace=.05, hspace=.05)        
    else:
        plt.subplots_adjust(wspace=.05, hspace=titles_hspace)
        
    if plot_title:
        plt.suptitle(plot_title)
        
    plt.show()

def show_single_with_alpha(image, alpha, figsize=(20, 20), plot_title=None, alpha_factor=0.3, title_size=4):
    plt.figure(figsize=figsize, dpi=600)

    plt.axis("off")
    plt.imshow(alpha, cmap='gray', aspect="auto")
    plt.imshow(image, aspect="auto", alpha=alpha_factor)
        
    if plot_title:
        plt.title(plot_title, fontsize=title_size, loc="left")
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    
def show_many_with_alpha(images, alphas, max_rows, max_cols, figsize=(20, 20), titles=None, titles_hspace=.2, plot_title=None, fontsize=2):
    plt.figure(figsize=figsize, dpi=300)

    for idx, (image, alpha) in enumerate(zip(images, alphas)):
        row = idx // max_cols
        col = idx % max_cols

        ax = plt.subplot2grid((max_rows, max_cols), (row, col))
        if titles != None:
            subtitle_conf = {"fontsize": fontsize}
            ax.set_title(titles[idx], fontdict=subtitle_conf, loc="left", pad=2)
        ax.axis("off")
        ax.imshow(alpha, cmap='gray', aspect="auto")
        ax.imshow(image, aspect="auto", alpha=0.3)
        
    if titles == None:
        plt.subplots_adjust(wspace=.05, hspace=.05)        
    else:
        plt.subplots_adjust(wspace=.05, hspace=titles_hspace)
        
    if plot_title:
        plt.suptitle(plot_title)
        
    plt.show()


def upscale_attention(attention, target_shape):
    attention_shape = np.shape(attention)
    assert len(attention_shape) == 2 
    sqr_shape = np.sqrt(attention_shape[1]).astype("uint8")
    attention = np.reshape(attention, (-1, sqr_shape, sqr_shape, 1))
    if len(target_shape) == 3:
        target_shape = (target_shape[0], target_shape[1])
    attention = [array_to_img(a).resize(target_shape, resample=NEAREST) for a in attention]
    # attention = [a.filter(ImageFilter.GaussianBlur(radius=5)) for a in attention]
    attention = np.array([img_to_array(a) for a in attention])
    attention = attention.astype("uint8")
    attention = np.squeeze(attention, axis=3)
    return attention


def visualize_images_with_caption_by_image_ids(config, sample_image_ids, sample_results, image_rows, image_cols, split_name):
    
    directory_path = "/".join([config.getDatasetImagesDirectoryPath(), split_name])
    image_infix = get_infix_from_config(config, split_name)
    captions = to_caption_listing_by_image_id(load_prepared_captions_json_from_config(config, split_name))
    
    provider = MscocoFileSystemImageProvider(directory_path, prefix="COCO_" + image_infix, vgg_like=False, target_size=(448, 448))
    sample_images = provider.get_images_for_image_ids(sample_image_ids)
    sample_images = sample_images.astype('uint8')
    
    titles = ["{:d}: {}".format(image_id, captions[image_id][0]) for image_id in sample_image_ids]
    titles = ["\n".join([titles[idx], r]) for idx, r in enumerate(sample_results)]
    show_many(sample_images, image_rows, image_cols, figsize=(4, 3), titles=titles)
