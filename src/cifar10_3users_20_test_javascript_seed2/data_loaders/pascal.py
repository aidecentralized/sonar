import xml.etree.ElementTree as ET
import torch
import torch.nn as nn

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# pascal dataset found here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#voc2012vs2011


def xml2label(xml_file):
    classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    yolo_data = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes:
            continue

        class_index = classes.index(class_name)

        # Extract bounding box coordinates
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_data.append([x_center, y_center, width, height, class_index])

    return yolo_data


# Defining a function to calculate Intersection over Union (IoU)
def iou(box1, box2, is_pred=True):
    if is_pred:
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format

        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Calculate the union area
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        # Calculate the IoU score
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        # Return IoU score
        return iou_score

    else:
        # IoU score based on width and height of bounding boxes

        # Calculate intersection area
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(
            box1[..., 1], box2[..., 1]
        )

        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score = intersection_area / union_area

        # Return IoU score
        return iou_score


# Non-maximum suppression function to remove overlapping bounding boxes
def nms(bboxes, iou_threshold, threshold):
    # Filter out bounding boxes with confidence below the threshold.
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort the bounding boxes by confidence in descending order.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # Initialize the list of bounding boxes after non-maximum suppression.
    bboxes_nms = []

    while bboxes:
        # Get the first bounding box.
        first_box = bboxes.pop(0)

        # Iterate over the remaining bounding boxes.
        for box in bboxes:
            # If the bounding boxes do not overlap or if the first bounding box has
            # a higher confidence, then add the second bounding box to the list of
            # bounding boxes after non-maximum suppression.
            if (
                box[0] != first_box[0]
                or iou(
                    torch.tensor(first_box[2:]),
                    torch.tensor(box[2:]),
                )
                < iou_threshold
            ):
                # Check if box is not in bboxes_nms
                if box not in bboxes_nms:
                    # Add box to bboxes_nms
                    bboxes_nms.append(box)

    # Return bounding boxes after non-maximum suppression.
    return bboxes_nms


# Function to convert cells to bounding boxes
def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    # Batch size used on predictions
    batch_size = predictions.shape[0]
    # Number of anchors
    num_anchors = len(anchors)
    # List of all the predictions
    box_predictions = predictions[..., 1:5]

    # If the input is predictions then we will pass the x and y coordinate
    # through sigmoid function and width and height to exponent function and
    # calculate the score and best class.
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    # Else we will just calculate scores and best class.
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    # Calculate cell indices
    cell_indices = (
        torch.arange(s)
        .repeat(predictions.shape[0], 3, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    # Calculate x, y, width and height with proper scaling
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    width_height = 1 / s * box_predictions[..., 2:4]

    # Concatinating the values and reshaping them in
    # (BATCH_SIZE, num_anchors * S * S, 6) shape
    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 6)

    # Returning the reshaped and converted bounding box list
    return converted_bboxes.tolist()


# Create a dataset class to load the images and labels from the folder
class Pascal(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        label_dir,
        anchors,
        image_size=416,
        grid_sizes=[13, 26, 52],
        num_classes=20,
        transform=None,
    ):

        # Read the csv file with image names
        self.label_list = pd.read_csv(csv_file)
        # Image and label directories
        self.image_dir = image_dir
        self.label_dir = label_dir
        # Image size
        self.image_size = image_size
        # Transformations
        self.transform = transform
        # Grid sizes for each scale
        self.grid_sizes = grid_sizes
        # Anchor boxes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        # Number of anchor boxes
        self.num_anchors = self.anchors.shape[0]
        # Number of anchor boxes per scale
        self.num_anchors_per_scale = self.num_anchors // 3
        # Number of classes
        self.num_classes = num_classes
        # Ignore IoU threshold
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 0] + ".xml")
        bboxes = xml2label(label_path)

        # Getting the image path
        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0] + ".jpg")
        image = np.array(Image.open(img_path).convert("RGB"))

        # Albumentations augmentations
        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # target : [probabilities, x, y, width, height, class_label]
        targets = [
            torch.zeros((self.num_anchors_per_scale, s, s, 6)) for s in self.grid_sizes
        ]

        # Identify anchor box and cell for each bounding box
        for box in bboxes:
            # Calculate iou of bounding box with anchor boxes
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False)
            # Selecting the best anchor box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box

            # At each scale, assigning the bounding box to the
            # best matching anchor box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                # Identifying the grid size for the scale
                s = self.grid_sizes[scale_idx]

                # Identifying the cell to which the bounding box belongs
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # Check if the anchor box is already assigned
                if not anchor_taken and not has_anchor[scale_idx]:

                    # Set the probability to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding box relative
                    # to the cell
                    x_cell, y_cell = s * x - j, s * y - i

                    # Calculating the width and height of the bounding box
                    # relative to the cell
                    width_cell, height_cell = (width * s, height * s)

                    # Idnetify the box coordinates
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    # Assigning the box coordinates to the target
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates

                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                    # Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the
                # IoU is greater than the threshold
                elif (
                    not anchor_taken
                    and iou_anchors[anchor_idx] > self.ignore_iou_thresh
                ):
                    # Set the probability to -1 to ignore the anchor box
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Return the image and the target
        return image, tuple(targets)


class PascalDataset:
    """
    Pascal Dataset Class.
    """

    def __init__(self, dpath: str) -> None:
        self.image_size = 416
        self.grid_sizes = [13, 26, 52]

        classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        self.num_cls = len(classes)
        self.num_channels = 3

        # Transform for training
        train_transform = A.Compose(
            [
                # Rescale an image so that maximum side is equal to image_size
                A.LongestMaxSize(max_size=self.image_size),
                # Pad remaining areas with zeros
                A.PadIfNeeded(
                    min_height=self.image_size,
                    min_width=self.image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Random color jittering
                A.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5
                ),
                # Flip the image horizontally
                A.HorizontalFlip(p=0.5),
                # Normalize the image
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
                # Convert the image to PyTorch tensor
                ToTensorV2(),
            ],
            # Augmentation for bounding boxes
            bbox_params=A.BboxParams(
                format="yolo", min_visibility=0.4, label_fields=[]
            ),
        )

        # Transform for testing
        test_transform = A.Compose(
            [
                # Rescale an image so that maximum side is equal to image_size
                A.LongestMaxSize(max_size=self.image_size),
                # Pad remaining areas with zeros
                A.PadIfNeeded(
                    min_height=self.image_size,
                    min_width=self.image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Normalize the image
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
                # Convert the image to PyTorch tensor
                ToTensorV2(),
            ],
            # Augmentation for bounding boxes
            bbox_params=A.BboxParams(
                format="yolo", min_visibility=0.4, label_fields=[]
            ),
        )

        # TODO: what to do about domain name?
        # csv_file = f'VOCdevkit/VOC2012/ImageSets/Main/{self.domain_name}_train.txt'
        ANCHORS = [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        ]
        self.train_dset = Pascal(
            csv_file=f"{dpath}ImageSets/Main/train.txt",
            image_dir=f"{dpath}JPEGImages/",
            label_dir=f"{dpath}Annotations/",
            image_size=self.image_size,
            grid_sizes=self.grid_sizes,
            num_classes=self.num_cls,
            transform=train_transform,
            anchors=ANCHORS,
        )

        self.test_dset = Pascal(
            csv_file=f"{dpath}ImageSets/Main/val_smol.txt",
            image_dir=f"{dpath}JPEGImages/",
            label_dir=f"{dpath}Annotations/",
            image_size=self.image_size,
            grid_sizes=self.grid_sizes,
            num_classes=self.num_cls,
            transform=test_transform,
            anchors=ANCHORS,
        )
