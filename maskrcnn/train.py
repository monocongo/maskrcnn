import argparse
import os
import random
from typing import Dict, List

from imgaug import augmenters as iaa
from mrcnn import model as modellib
from mrcnn.config import Config

from maskrcnn.dataset import MaskrcnnMasksDataset, MaskrcnnViajsonDataset


# ------------------------------------------------------------------------------
class MaskrcnnConfig(Config):

    def __init__(
            self,
            config_name: str,
            class_names: Dict,
            train_indices: List[int],
            valid_indices: List[int],
            gpu_count: int = 1,
            images_per_gpu: int = 1,
    ):
        """
        Constructor function used to initialize objects of this class.

        :param config_name: arbitrary name to use for the configuration
        :param class_names: dictionary of class IDs to class labels
        :param train_indices: indices of the image file paths list to use for training
        :param valid_indices: indices of the image file paths list to use for validation
        :param gpu_count: the number of GPUs to use for training
        :param images_per_gpu: the numner of images to use per GPU
        """

        # number of classes (+1 for the background)
        # NOTE: this needs to be set before calling the parent constructor, as
        # described here: https://github.com/matterport/Mask_RCNN/issues/410#issuecomment-382364349
        self.NUM_CLASSES = len(class_names) + 1

        # call the parent constructor
        super().__init__()

        # give the configuration a recognizable name
        self.NAME = config_name

        # set the number of GPUs to use training along with the number of
        # images per GPU (which may have to be tuned depending on how
        # much memory your GPU has)
        self.GPU_COUNT = gpu_count
        self.IMAGES_PER_GPU = images_per_gpu

        # set the number of steps per training epoch and validation cycle
        batch_size = gpu_count * images_per_gpu
        self.STEPS_PER_EPOCH = len(train_indices) // batch_size
        self.VALIDATION_STEPS = len(valid_indices) // batch_size


# ------------------------------------------------------------------------------
def class_ids_to_labels(
        labels_path: str,
) -> Dict:
    """
    Reads a text file, which is assumed to contain one class label per line, and
    returns a dictionary with integer keys (starting with 1) mapped to the class
    label that was found at the key's line number.

    So a labels file like so:

    cat
    dog
    panda

    will result in a dictionary like so:

    {
      1: "cat,
      2: "dog",
      3: "panda",
    }

    :param labels_path: path to a file containing class labels used in
        a segmentation dataset, with one class label per line
    :return: dictionary mapping ID values to corresponding labels
    """

    class_labels = {}
    with open(labels_path, "r") as class_labels_file:
        class_id = 1
        for class_label in class_labels_file:
            class_labels[class_id] = class_label.strip()
            class_id += 1

    return class_labels


# ------------------------------------------------------------------------------
def train_model(
        images_dir: str,
        pretrained_model: str,
        output_dir: str,
        classes: str,
        epochs: int = 40,
        train_split: float = 0.8,
        masks_dir: str = None,
        masks_suffix: str = None,
        viajson: str = None,
):
    """
    Trains (fine tuning) the Mask R-CNN model using a training dataset and
    pre-trained model weights.

    :param images_dir: directory containing image files
    :param pretrained_model: model weights for pre-trained model
    :param output_dir: directory where log files and model weights will be written
    :param classes: path to text file containing the class labels used in the
        dataset, with one class label per line
    :param epochs: number of training iterations
    :param train_split: percentage of images to use for training (remainder
        for validation)
    :param masks_dir: directory containing mask files
    :param masks_suffix: mask file suffix, assumes mask file names equal the file
        ID the corresponding image file (file name minus .jpg extension) plus
        this suffix
    :param viajson: JSON annotations file created by the VGG Image Annotator tool
    """

    # get list of image paths (all JPG images in the images directory)
    image_paths = []
    for file_name in os.listdir(images_dir):
        if file_name.endswith('.jpg'):
            image_paths.append(os.path.join(images_dir, file_name))

    # get a list of indices for training and validation
    # images by randomizing the image paths list indices
    # and slicing according to the training split percentage
    path_indices = list(range(0, len(image_paths)))
    random.seed(42)
    random.shuffle(path_indices)
    i = int(len(path_indices) * train_split)
    train_indices = path_indices[:i]
    valid_indices = path_indices[i:]

    # read the classes file to get the class IDs mapped to class labels/names
    class_ids = class_ids_to_labels(classes)

    # for each class ID add the mask value used in the mask image
    # TODO the values below are specific to mask files where there
    #  is a single class with 255 as the grayscale pixel mask value
    #  -- modify so we can specify the mask pixel value(s) used for each class ID
    class_masks = {  # class IDs to mask grayscale pixel values
        1: 255,  # class ID 1 maps to mask image grayscale pixel value 255
    }

    # load the training dataset
    if viajson:
        train_dataset = MaskrcnnViajsonDataset(image_paths, class_ids, viajson)
    else:
        train_dataset = MaskrcnnMasksDataset(image_paths, class_ids, class_masks, masks_dir, masks_suffix)
    train_dataset.add_images("maskrcnn", train_indices)
    train_dataset.prepare()

    # load the validation dataset
    if viajson:
        valid_dataset = MaskrcnnViajsonDataset(image_paths, class_ids, viajson)
    else:
        valid_dataset = MaskrcnnMasksDataset(image_paths, class_ids, class_masks, masks_dir, masks_suffix)
    valid_dataset.add_images("maskrcnn", valid_indices)
    valid_dataset.prepare()

    # initialize the image augmentation process
    augmentation = iaa.SomeOf(
        (0, 2),
        [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Affine(rotate=(-10, 10))],
    )

    # initialize the training configuration
    config = MaskrcnnConfig("maskrcnn", class_ids, train_indices, valid_indices)

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
        epochs=(epochs - layer_heads_training_epochs),
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
        required=False,
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
        required=False,
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
    args_parser.add_argument(
        "--viajson",
        required=False,
        type=str,
        help="path of the annotations JSON file created by the VGG Image "
             "Annotator (VIA) tool, representing the masks for the images",
    )
    args = vars(args_parser.parse_args())

    # train the model
    train_model(
        args["images"],
        args["pretrained"],
        args["output"],
        args["classes"],
        args["epochs"],
        args["train_split"],
        args["masks"],
        args["masks_suffix"],
        args["viajson"],
    )


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage: 
    $ python train.py --images ~/datasets/handgun/images \
        --masks ~/datasets/handgun/masks \
        --masks_suffix _segmentation.jpg
        --pretrained ~/maskrcnn/weights/mask_rcnn_coco.h5 \
        --output ~/maskrcnn/weights/mask_rcnn_handgun.h5 \
        --classes ~/datasets/handgun/class_labels.txt \
        --epochs 50 --train_split 0.8 
    """

    main()
