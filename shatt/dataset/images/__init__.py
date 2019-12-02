import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from shatt.dataset import determine_file_path
import numpy as np
from shatt import SPLIT_TRAIN, SPLIT_VALIDATE, SPLIT_TEST_DEV, SPLIT_TEST
from PIL import ImageDraw, Image


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def draw_box_on_image(draw, box, fill=False, scale_factors=(1,1)):
    x0, y0, width, height = box
    
    """ perform scaling if necessary """
    scale_x = scale_factors[0]
    scale_y = scale_factors[1]
    x0 = x0 * scale_x
    y0 = y0 * scale_y
    width = width * scale_x
    height = height * scale_y
    
    """ the actual drawing """
    p0 = x0, y0
    p1 = x0 + width, y0 + height
    if fill:
        draw.rectangle([p0, p1], fill="white", width=0)
    else:
        #draw.rectangle([p0, p1], outline="red", width=3)
        draw.rectangle([p0, p1], outline="orange", width=10)


def draw_boxes_on_image(boxes, image, fill=False, scale_factors=(1,1)):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw_box_on_image(draw, box["box"], fill, scale_factors)


def draw_boxes_on_images(images, boxes, fill=False, scale_factors=(1,1)):
    results = []
    for idx, image in enumerate(images):
        image = image.astype("uint8")
        with Image.fromarray(image) as im:
            draw_boxes_on_image(boxes[idx], im, fill, scale_factors)
            image = np.array(im)
            results.append(image)
    return np.array(results)


def store_numpy_to(data, target_directory_path_or_file, lookup_file_name=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(target_directory_path_or_file, lookup_file_name, to_read=False)    
    
    with open(file_path, "wb") as f:
        np.save(f, data)
    return file_path


def load_numpy_from(directory_or_file, lookup_filename=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename)
    
    with open(file_path, "rb") as f:
        data = np.load(f)
    return data


def get_infix_from_config(config, split_name):
    if split_name == SPLIT_TRAIN:
        return config.getTrainingImageNameInfix()
    if split_name == SPLIT_VALIDATE:
        return config.getValidationImageNameInfix()
    if split_name in [SPLIT_TEST, SPLIT_TEST_DEV]:
        return config.getTestImageNameInfix()
    raise Exception("Cannot determine image name infix for split " + split_name)

    
def _exists_image_path_by_id(image_prefix, image_id, directory_path=None, file_ending="jpg"):
    file_path = to_image_path_by_id(image_prefix, image_id, directory_path, file_ending)
    return os.path.isfile(file_path)

    
def to_image_path_by_id(image_prefix, image_id, directory_path=None, file_ending="jpg"):
    """
        Returns MSCOCO naming pattern e.g. COCO_<split_name>_<image_id>
        
        For example COCO_train2014_000000000009.jpg
    """
    file_name = "{}_{:012}.{}".format(image_prefix, image_id, file_ending)
    if directory_path:
        return "/".join([directory_path, file_name])
    return file_name

        
def resize_images(images, target_size):
    results = []
    for image in images:
        image = image.astype("uint8")
        with Image.fromarray(image) as im:
            """ WARNING: NOT USING RESIZE LIKE KERAS DOES MIGHT CAUSE DAMAGE """
            im = im.resize(target_size)
            results.append(np.array(im))
    return results


def _get_image(image_file_path, target_shape):
    with load_img(image_file_path, target_size=target_shape) as image:
        imagearr = img_to_array(image)
    return imagearr


def get_image(image_file_path):
    with load_img(image_file_path) as image:
        imagearr = img_to_array(image)
        return imagearr


def get_image_paths(directory_path, file_ending=".jpg"):
    return ["/".join([directory_path, file]) for file in os.listdir(directory_path) if file.endswith(file_ending)]


def extract_image_id(image_path):
    """
        Returns the image id from a MSCOCO naming pattern e.g. COCO_<split_name>_<image_id>
        
        For example COCO_train2014_000000000009.jpg has the image id 9
    """
    filename = os.path.splitext(image_path)[0]
    image_id = filename.split("_")[2]
    image_id = int(image_id)
    return image_id


def extract_to_image_ids_ordered(image_paths):
    """ Its critical to keep the order here """
    return np.array([extract_image_id(image_path) for image_path in image_paths])

