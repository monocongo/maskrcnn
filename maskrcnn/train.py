import argparse
import os
import random
from typing import Dict, List

from imgaug import augmenters as iaa
from mrcnn import model as modellib
from mrcnn.config import Config

from maskrcnn.dataset import MaskrcnnDataset


# ------------------------------------------------------------------------------
class MaskrcnnConfig(Config):

    def __init__(
            self,
            config_name: str,
            class_names: Dict,
            train_indices: List[int],
            valid_indices: List[int],
    ):
        """
        Constructor function used to initialize objects of this class.

        :param config_name:
        :param class_names:
        :param train_indices:
        :param valid_indices:
        """

        # call the parent constructor
        super().__init__()

        # give the configuration a recognizable name
        self.NAME = config_name

        # set the number of GPUs to use training along with the number of
        # images per GPU (which may have to be tuned depending on how
        # much memory your GPU has)
        gpu_count = 1
        images_per_gpu = 1

        # set the number of steps per training epoch and validation cycle
        self.STEPS_PER_EPOCH = len(train_indices) // (images_per_gpu * gpu_count)
        self.VALIDATION_STEPS = len(valid_indices) // (images_per_gpu * gpu_count)

        # number of classes (+1 for the background)
        self.NUM_CLASSES = len(class_names) + 1


# ------------------------------------------------------------------------------
def train(
        images_dir: str,
        masks_dir: str,
        masks_suffix: str,
        pretrained_model: str,
        output_dir: str,
        classes: str,
        epochs: int = 40,
        train_split: float = 0.8,
):
    """
    TODO

    :param images_dir:
    :param masks_dir:
    :param masks_suffix:
    :param pretrained_model:
    :param output_dir:
    :param classes: path to text file containing the class labels used in the
        dataset, one per line
    :param epochs:
    :param train_split:
    :return:
    """

    # get list of image paths (all JPG images in the images directory)
    image_paths = []
    for file_name in os.listdir(images_dir):
        if file_name.endswith('.jpg'):
            image_paths.append(os.path.join(images_dir, file_name))

    # get a list of indices for training and validation
    # images by randomizing the image paths list indices
    # and slicing according to the training split percentage
    idxs = list(range(0, len(image_paths)))
    random.seed(42)
    random.shuffle(idxs)
    i = int(len(idxs) * train_split)
    train_indices = idxs[:i]
    valid_indices = idxs[i:]

    # read the classes file to get the class IDs mapped to class labels/names
    class_names = {}
    class_id = 1
    with open(classes, "r") as class_labels_file:
        for class_label in class_labels_file:
            class_names[class_id] = class_label
            class_id += 1

    # for each class ID add the mask value used in the mask image
    # TODO the values below are specific to mask files where there
    #  is a single class with 255 as the grayscale pixel mask value
    #  -- modify so we can specify the mask pixel value(s) used for each class ID
    class_masks = {  # class IDs to mask grayscale pixel values
        1: 255,  # class ID 1 maps to mask image grayscale pixel value 255
    }

    # load the training dataset
    train_dataset = MaskrcnnDataset(image_paths, class_names, class_masks, masks_dir, masks_suffix)
    train_dataset.add_images("maskrcnn", train_indices)
    train_dataset.prepare()

    # load the validation dataset
    valid_dataset = MaskrcnnDataset(image_paths, class_names, class_masks, masks_dir, masks_suffix)
    valid_dataset.add_images("maskrcnn", valid_indices)
    valid_dataset.prepare()

    # initialize the image augmentation process
    augmentation = iaa.SomeOf(
        (0, 2),
        [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Affine(rotate=(-10, 10))],
    )

    # initialize the training configuration
    config = MaskrcnnConfig("maskrcnn", class_names, train_indices, valid_indices)

    # initialize the model and load the pre-trained
    # weights we'll use to perform fine-tuning
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=output_dir)
    model.load_weights(
        pretrained_model,
        by_name=True,
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
    )

    # train only the layer heads
    layer_heads_training_epochs = 20
    model.train(
        train_dataset,
        valid_dataset,
        epochs=layer_heads_training_epochs,
        layers="heads",
        learning_rate=config.LEARNING_RATE,
        augmentation=augmentation,
    )

    # unfreeze the body of the network and train all layers
    model.train(
        train_dataset,
        valid_dataset,
        epochs=epochs,
        layers="all",
        learning_rate=config.LEARNING_RATE / 10,
        augmentation=augmentation,
    )


# ------------------------------------------------------------------------------
def main():

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="directory path containing input image files",
    )
    args_parser.add_argument(
        "--masks",
        required=True,
        type=str,
        help="directory path containing image masks corresponding "
             "to the image files in the images directory",
    )
    args_parser.add_argument(
        "--pretrained",
        required=True,
        type=str,
        help="path of the pretrained model file that will serve as "
             "the starting point for the trained (fine tuned) model",
    )
    args_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="directory path where all output model snapshots and logs will "
             "be stored, including the final trained (fine tuned) model file",
    )
    args_parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=40,
        help="number of training iterations",
    )
    args_parser.add_argument(
        "--train_split",
        type=float,
        required=False,
        default=0.8,
        help="percentage of the training data to use for training "
             "((1.0 - this value) will be used for validation)",
    )
    args_parser.add_argument(
        "--masks_suffix",
        required=True,
        type=str,
        help="each mask file name under the masks directory is assumed to be "
             "composed of the the image file ID plus this suffix, for example "
             "<FILE_ID>_segmentation.jpg, with \"_segmentation.jpg\" as "
             "the suffix",
    )
    args_parser.add_argument(
        "--classes",
        required=True,
        type=str,
        help="path of the class labels file listing one class per line",
    )

    args = vars(args_parser.parse_args())

    # train the model
    train(
        args["images"],
        args["masks"],
        args["masks_suffix"],
        args["pretrained"],
        args["output"],
        args["epochs"],
        args["classes"],
        args["train_split"],
    )


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage: 
    $ python train.py --images ~/datasets/handgun/images \
        --masks ~/datasets/handgun/masks \
        --pretrained ~/maskrcnn/weights/mask_rcnn_coco.h5 \
        --output ~/maskrcnn/weights/mask_rcnn_handgun.h5 \
        --epochs 50 --train_split 0.8 
    """

    main()
