
DEFAULT_TFRECORD_FILE_NAME = "shatt_images.tfrecord"


def get_tfrecord_filename(split_name=None):
    if split_name:
        return "shatt_images_{}.tfrecord".format(split_name)
    return DEFAULT_TFRECORD_FILE_NAME
