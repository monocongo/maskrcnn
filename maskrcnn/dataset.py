import json
import os
from typing import Dict, List

import cv2
from cvdata.utils import image_dimensions
import imutils
from mrcnn.utils import Dataset
import numpy as np


class MaskrcnnDataset(Dataset):

    def __init__(
            self,
            image_paths: List[str],
            class_names: Dict,
            width: int = 1024,
            reference: str = None,
    ):
        """
        Constructor function used to initialize objects of this class.

        :param image_paths: list of image paths
        :param class_names: dictionary of class IDs to labels
        :param width: width to which images and masks will eventually be resized
        :param reference: description, URL, etc. providing some reference for
            the source or other details of the image
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

        # reference of images/masks
        self.reference = reference

    def add_images(
            self,
            data_source: str,
            idxs: List[int],
    ):
        """
        Add images into the dataset based on indices of the image file paths list.

        :param data_source:
        :param idxs: indices of the file paths list for the images be added
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
    ) -> np.ndarray:
        """
        Gets an array of RGB pixel values, resized to the dataset's specified
        width (with aspect ratio preserved), and with shape (height, width, 3).

        :param image_id:
        :return: 3-D numpy array, with shape (height, width, 3)
        """

        # grab the image path, load it, and convert
        # it from BGR to RGB color channel ordering
        image_path = self.image_info[image_id]["path"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize the image, preserving the aspect ratio
        image = imutils.resize(image, width=self.width)

        # return the image
        return image

    def image_reference(
            self,
            image_id: str,
    ) -> str:
        """
        Return a link to the image in its source website or details about
        the image that help looking it up or debugging it.

        :param image_id: file ID for the image
        :return: string with reference information about the source or other details
        """

        return self.reference


class MaskrcnnViajsonDataset(MaskrcnnDataset):

    def __init__(
            self,
            image_paths: List[str],
            class_names: Dict,
            viajson: str,
            width: int = 1024,
            reference: str = None,
    ):
        """
        Constructor function used to initialize objects of this class.

        :param image_paths:
        :param class_names: dictionary of class IDs to labels
        :param viajson: path to VIA annotations JSON file containing segmentation
            regions defining the masks
        :param width:
        :param reference:
        """

        # call the parent constructor
        super().__init__(self, image_paths, class_names, width, reference)

        self.via_annotations = self.load_annotation_data(viajson)

    @staticmethod
    def load_annotation_data(
            viajson_file_path: str,
    ) -> Dict:
        """
        Gets the annotations data for the images in the dataset from an
        annotation JSON file created by the VGG Image Annotator (VIA) tool.

        :param viajson_file_path: annotation JSON file created using the VIA tool
        :return: a dictionary with the image file names as keys and annotation
            data as values
        """

        # load the contents of the annotation JSON file (created
        # using the VIA tool) and initialize the annotations
        # dictionary
        annotations = json.loads(open(viajson_file_path).read())
        image_annotations = {}

        # loop over the file ID and annotations themselves (values)
        for (fileID, data) in sorted(annotations.items()):
            # store the data in the dictionary using the filename as the key
            image_annotations[data["filename"]] = data

        # return the annotations dictionary
        return image_annotations

    def load_mask(
            self,
            image_id: str,
    ) -> (np.ndarray, List):

        # grab the image info and then grab the annotation data for
        # the current image based on the unique image ID
        info = self.image_info[image_id]
        annotation = self.via_annotations[info["id"]]

        # get the image's dimensions
        width, height, _ = image_dimensions(info["path"])

        # allocate memory for our [height, width, num_instances] 3-D array
        # where each "instance" (region) effectively has its own "channel"
        num_instances = len(annotation["regions"])
        masks = np.zeros(shape=(height, width, num_instances), dtype="uint8")

        # allocate memory for our [num_instances] 1-D array to contain
        # the class IDs corresponding to each mask instance
        mask_class_ids = np.full(shape=(num_instances,), dtype="int32", fill_value=-1)

        # loop over each of the annotated regions
        for (i, region) in enumerate(annotation["regions"]):

            # allocate memory for the region mask
            region_mask = np.zeros(masks.shape[:2], dtype="uint8")

            # grab the shape and region attributes
            shape_attributes = region["shape_attributes"]
            region_attributes = region["region_attributes"]

            # find the class ID corresponding to the region's class attribute
            class_label = region_attributes["class"]
            class_id = -1
            for key, label in self.class_names.items():
                if label == class_label:
                    class_id = key
                    break
            if class_id == -1:
                raise ValueError(
                    "No corresponding class ID found for the class label "
                    f"found in the region attributes -- label: {class_label}",
                )

            # get the array of (x, y)-coordinates for the region's mask polygon
            x_coords = shape_attributes["all_points_x"]
            y_coords = shape_attributes["all_points_y"]
            coords = zip(x_coords, y_coords)
            poly_coords = [[x, y] for x, y in coords]
            pts = np.array(poly_coords, np.int32)

            # reshape the points to (<# of coordinates>, 1, 2)
            pts = pts.reshape((-1, 1, 2))

            # # draw the polygon mask, using the class ID as the mask value
            # grayscale_rgb = [class_id]*3
            # cv2.polylines(region_mask, [pts], True, grayscale_rgb)

            # draw the polygon mask, using the class ID as the mask value
            cv2.fillPoly(region_mask, pts, np.uint8(class_id))

            # store the mask in the masks array
            masks[:, :, i] = region_mask

            # store the class ID for this channel (mask region)
            mask_class_ids[i] = class_id

        # return the mask array and array of mask class IDs
        return masks.astype("bool"), mask_class_ids


class MaskrcnnMasksDataset(MaskrcnnDataset):

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

        # call the parent constructor
        super().__init__(self, image_paths, class_names, width, reference)

        # directory and suffix of mask file paths
        # mask files should share the ID of their corresponding image,
        # and will have a path == masks_dir + os.sep + image_id + masks_suffix
        self.masks_dir = masks_dir
        self.masks_suffix = masks_suffix

        # dictionary mapping class IDs to the grayscale pixel
        # value used for masks of objects of the class
        self.class_masks = class_masks

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
