import argparse
import os
import random


# ------------------------------------------------------------------------------
def train(
        images_dir: str,
        masks_dir: str,
        pretrained_model: str,
        output_model: str,
        epochs: int = 40,
        train_split: float = 0.8,
):
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

    # TODO
    pass


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
        help="path of the resulting trained (fine tuned) model file",
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
    args = vars(args_parser.parse_args())

    train(
        args["images"],
        args["masks"],
        args["pretrained"],
        args["output"],
        args["epochs"],
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
