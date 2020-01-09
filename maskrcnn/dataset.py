import os
from typing import Dict, List

import cv2
import imutils
from mrcnn.utils import Dataset
import numpy as np


class MaskrcnnDataset(Dataset):

    def __init__(
            self,
            image_paths: List[str],
            class_names: Dict,
            class_masks: Dict,
            masks_dir: str,
            masks_suffix: str,
            width: int = 1024,
            reference: str = None,
    ):
        """
        Constructor function used to initialize objects of this class.

        :param image_paths:
        :param class_names: dictionary of class IDs to labels
        :param class_masks: dictionary of class IDs to mask pixel values
        :param masks_dir:
        :param masks_suffix:
        :param width:
        :param reference:
        """

        # validate arguments
        if width <= 0:
            raise ValueError("Invalid width argument: must be greater than zero")

        # call the parent constructor
        super().__init__(self)

        # paths to individual images
        self.image_paths = image_paths

        # dictionary mapping class IDs to class labels (names)
        self.class_names = class_names

        # the width dimension to which all images will be resized
        self.width = width

        # directory and suffix of mask file paths
        # mask files should share the ID of their corresponding image,
        # and will have a path == masks_dir + os.sep + image_id + masks_suffix
        self.masks_dir = masks_dir
        self.masks_suffix = masks_suffix

        # dictionary mapping class IDs to the grayscale pixel
        # value used for masks of objects of the class
        self.class_masks = class_masks

        # reference of images/masks
        self.reference = reference

    def load_images(
            self,
            data_source: str,
            idxs: List[int],
    ):
        """

        :param data_source:
        :param idxs:
        :return:
        """

        # loop over each of the class IDs/names and add to the source dataset
        for (class_id, label) in self.class_names.items():
            self.add_class(data_source, class_id, label)

        # loop over the image path indexes
        for i in idxs:
            # extract the image filename to serve as the unique image ID
            image_path = self.image_paths[i]
            filename = image_path.split(os.path.sep)[-1]

            # add the image to the dataset
            self.add_image(data_source, image_id=filename, path=image_path)

    def load_image(
            self,
            image_id: str,
    ):
        """

        :param image_id:
        :return:
        """

        # grab the image path, load it, and convert it
        #  from BGR to RGB color channel ordering
        image_path = self.image_info[image_id]["path"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize the image, preserving the aspect ratio
        image = imutils.resize(image, width=self.width)

        # return the image
        return image

    def load_mask(
            self,
            image_id: str,
    ) -> (np.ndarray, List):
        """
        Loads an array of masks from a mask file corresponding to an image ID.

        :param image_id: image ID, should correspond to a mask file in our masks
            directory
        :return: a tuple composed of 1) a 3-D array of masks, with the third
            dimension the number of classes, and 2) a list of class IDs
            corresponding to the third dimension of the masks array
        """

        # grab the image info and derive the full annotation file path
        file_id = self.image_info[image_id]["id"].split(".")[0]
        mask_path = os.path.join(self.masks_dir, f"{file_id}{self.masks_suffix}")

        # load the annotation mask and resize it using nearest neighbor interpolation
        mask = cv2.imread(mask_path)
        mask = cv2.split(mask)[0]
        mask = imutils.resize(mask, width=self.width, inter=cv2.INTER_NEAREST)

        # set the mask values to their corresponding class IDs
        for (class_id, mask_value) in self.class_masks.items():
            mask[mask == mask_value] = class_id

        # determine the number of unique class labels in the mask
        class_ids = np.unique(mask)

        # the class ID with value '0' is actually the background
        # which we should ignore and remove from the unique set of
        # class identifiers
        class_ids = np.delete(class_ids, [0])

        # allocate memory for our [height, width, num_instances]
        # array where each "instance" effectively has its own "channel"
        masks = np.zeros((mask.shape[0], mask.shape[1], len(class_ids)), dtype="uint8")

        # loop over the class IDs
        for (i, class_id) in enumerate(class_ids):

            # construct a mask for *only* the current label
            class_mask = np.zeros(mask.shape, dtype="uint8")
            class_mask[mask == class_id] = 1

            # store the class mask in the masks array
            masks[:, :, i] = class_mask

        # return the mask array and class IDs
        return masks.astype("bool"), class_ids.astype("int32")

    def image_reference(self, image_id):
        """
        Return a link to the image in its source website or details about
        the image that help looking it up or debugging it.

        :param image_id: file ID for the image
        :return: string with reference information about the source or other details
        """

        return self.reference
