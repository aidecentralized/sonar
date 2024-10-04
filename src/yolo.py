import torch
import torch.nn as nn
import torch.optim as optim
import os

from tqdm import tqdm

# implemented following
# https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/


# Helpers
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


# Defining CNN Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, bias=not use_batch_norm, **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        # Applying convolution
        x = self.conv(x)
        # Applying BatchNorm and activation if needed
        if self.use_batch_norm:
            x = self.bn(x)
            return self.activation(x)
        else:
            return x


# Defining residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()

        # Defining all the layers in a list and adding them based on number of
        # repeats mentioned in the design
        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1),
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    # Defining forward pass
    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x = x + residual
        return x


# Defining scale prediction class
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Defining the layers in the network
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2 * in_channels, (num_classes + 5) * 3, kernel_size=1),
        )
        self.num_classes = num_classes

    # Defining the forward pass and reshaping the output to the desired output
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5)
    def forward(self, x):
        output = self.pred(x)
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        output = output.permute(0, 1, 3, 4, 2)
        return output


# Class for defining YOLOv3 model
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Layers list for YOLOv3
        self.layers = nn.ModuleList(
            [
                CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
                CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
                ResidualBlock(64, num_repeats=1),
                CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
                ResidualBlock(128, num_repeats=2),
                CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
                ResidualBlock(256, num_repeats=8),
                CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
                ResidualBlock(512, num_repeats=8),
                CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
                ResidualBlock(1024, num_repeats=4),
                CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
                CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
                ResidualBlock(1024, use_residual=False, num_repeats=1),
                CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
                ScalePrediction(512, num_classes=num_classes),
                CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=2),
                CNNBlock(768, 256, kernel_size=1, stride=1, padding=0),
                CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
                ResidualBlock(512, use_residual=False, num_repeats=1),
                CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
                ScalePrediction(256, num_classes=num_classes),
                CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=2),
                CNNBlock(384, 128, kernel_size=1, stride=1, padding=0),
                CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
                ResidualBlock(256, use_residual=False, num_repeats=1),
                CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
                ScalePrediction(128, num_classes=num_classes),
            ]
        )

    # Forward pass for YOLOv3 with route connections and scale predictions
    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs


# Defining YOLO loss class
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        # Identifying which cells in target have objects
        # and which have no objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = self.bce(
            (pred[..., 0:1][no_obj]),
            (target[..., 0:1][no_obj]),
        )

        # Reshaping anchors to match predictions
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        # Box prediction confidence
        box_preds = torch.cat(
            [self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * anchors], dim=-1
        )
        # Calculating intersection over union for prediction and target
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        # Calculating Object loss
        object_loss = self.mse(
            self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj]
        )

        # Predicted box coordinates
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        # Target box coordinates
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        # Calculating box coordinate loss
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])

        # Calculating class loss
        class_loss = self.cross_entropy(
            (pred[..., 5:][obj]), target[..., 5][obj].long()
        )

        # Total loss
        return box_loss + object_loss + no_object_loss + class_loss


# TODO: where should this go?
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save model variable
load_model = False
save_model = True


# Function to load checkpoint
def load_checkpoint(checkpoint_file, model):
    print("==> Loading checkpoint")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    # model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # optimizer.load_state_dict(checkpoint["optimizer"])

    # for param_group in optimizer.param_groups:
    # param_group["lr"] = lr


def yolo(pretrained=False, **kwargs):
    r"""Yolov3 implementation from https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    checkpoint_file = "./datasets/pascal/my_checkpoint.pth.tar"
    model = YOLOv3().to(device)
    if pretrained:
        print("loading model")
        load_checkpoint(checkpoint_file, model)
        print("model loaded")
    return model


# Define the train function to train the model
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    # Creating a progress bar
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = []

    # Iterating over the training data
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        with torch.cuda.amp.autocast(enabled=False):
            # Getting the model predictions
            outputs = model(x)
            # Calculating the loss at each scale
            loss = (
                loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2])
            )

        # Add the loss to the list
        losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        scaler.scale(loss).backward()

        # Optimization step
        scaler.step(optimizer)

        # Update the scaler for next iteration
        scaler.update()

        # update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)
