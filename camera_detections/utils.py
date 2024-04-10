from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import os

font = ImageFont.load_default()
string_labels = ["bike", "car", "cycle", "pedestrian", "signal"]


def validate_path(source: str) -> str:
    """
    This function checks if the path exists.

    Parameters
    ----------
        source: str
            This is the path to check
    
    Returns
    -------
        source: str
            The path is returned if it exists.

    Raises
    ------
        FileNotFoundError
            This error raised if the path does not exist.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"The path {source} does not exist.")
    return source

def resize(
    image: np.ndarray, 
    size: tuple=None, 
) -> np.ndarray:
    """
    Resizes a numpy image array with the specified size.

    Parameters
    ----------
        image: np.ndarray
            This is the image to resize.

        size: tuple
            (width, height) to resize the image.

    Returns
    -------
        image: np.ndarray
            This is the resized image.
    """    
    shape = image.shape

    # Keep the original shape.
    if size is None:
        size = (shape[1], shape[0])
    image = Image.fromarray((image * 1).astype(np.uint8)).convert("RGB")
    image = image.resize(size)
    return np.asarray(image)

def normalize(image:np.ndarray, normalization: str, input_type: str="float32") -> np.ndarray:
    """
    Perform image normalization primarily used for model input preprocessing. 
    Translation of values in between 0 and 1.

    Parameters
    ----------
        image: np.ndarray
            This is the image to normalize.

        normalization: str
            This is the type of image normalization to perform.

        input_type: str
            This is the input type of the model. By default it is a float32 model.

    Returns
    -------
        image: np.ndarray
            This is the normalized image.
    """
    if normalization.lower() == "signed":
        return np.expand_dims((image/127.5)-1.0, 0).astype(np.dtype(input_type))
    elif normalization.lower() == "unsigned":
        return np.expand_dims(image/255.0, 0).astype(np.dtype(input_type))
    else:
        return np.expand_dims(image, 0).astype(np.dtype(input_type))

def yolo2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    This converts yolo annotation format into pascalvoc.

    Parameters
    ----------
        boxes: np.ndarray
            These are the bounding boxes in yolo format.

    Returns
    -------
        boxes: np.ndarray
            The bounding boxes in pascalvoc format.
    """
    w_c = boxes[..., 2:3]
    h_c = boxes[..., 3:4]
    boxes[..., 0:1] = boxes[..., 0:1] - w_c/2
    boxes[..., 1:2] = boxes[..., 1:2] - h_c/2
    boxes[..., 2:3] = boxes[..., 0:1] + w_c
    boxes[..., 3:4] = boxes[..., 1:2] + h_c
    return boxes

def batch_iou(box1, box2, eps: float=1e-7):
    """
    Performs a batch IoU for Tflite NMS detections.

    Parameters
    ----------  
        box1: torch.Tensor 
            (N,4) tensors containing bounding boxes.

        box1: torch.Tensor 
            (N,4) tensors containing bounding boxes.

        eps: float
            A minimal value to prevent division by zero.

    Returns
    -------
        iou: torch.Tensor 
            This contains an array of IoUs for each bounding box pair provided.
    """
    (a1,a2), (b1,b2) = box1.unsqueeze(1).chunk(2,2), box2.unsqueeze(0).chunk(2,2) 
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2-a1).prod(2) + (b2-b1).prod(2) - inter + eps)

def draw_bounding_box(
        image_draw: ImageDraw.ImageDraw, 
        box: tuple,
        color="LimeGreen",
        width=3
    ):
    """
    Draw bounding boxes on the image.

    Parameters
    ----------
        image_draw: ImageDraw.ImageDraw
            This is the ImageDraw object initialized using Pillow by
            passing an PIL.Image.Image object.

        box: tuple
            This contains ((xmin, ymin), (xmax, ymax)) bounding box coordinates
            in pixels.

        color: str
            The color of the bounding box.

        width: int
            This is the width of the lines of the bounding box.
    """
    if (box[0][0] < box[1][0]) and (box[0][1] < box[1][1]):
        image_draw.rectangle(
            box,
            outline=color,
            width=width)
    
def draw_text(
        image_draw: ImageDraw.ImageDraw,
        text: str,
        position: tuple,
        color: str="black", 
        align: str="left"
    ):
    """
    Draws text on the image.

    Parameters
    ----------
        image_draw: ImageDraw.ImageDraw
            This is the ImageDraw object initialized using Pillow by
            passing an PIL.Image.Image object.

        text: str
            This is the text to draw on the image.

        position: tuple
            This is the (xmin, ymin) position to place the text on the image.

        color: str
            This is the color of the text.

        align: str
            This is the text alignment on the image.
    """
    image_draw.text(
        position,
        text,
        font=font,
        align=align,
        fill=color
    )

def visualize(
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
    """
    This method draws the shelving bounding boxes detected.

    Parameters
    ----------
        image: np.ndarray
            The image to draw shelving bounding boxes.

        boxes: np.ndarray
            The detected bounding boxes to draw.

        scores: np.ndarray
            The scores of each bounding boxes.

        labels: np.ndarray
            The labels of each shelving.

    Returns
    -------
        image: np.ndarray
            The image with drawn bounding boxes.
    """
    image_drawn = Image.fromarray(image)
    image_draw = ImageDraw.Draw(image_drawn)
    
    for box, score, label in zip(boxes, scores, labels):
        draw_bounding_box(
            image_draw, ((box[0], box[1]), (box[2], box[3])))

        text = f"{string_labels[int(label)]} {round(score*100, 2)}"
        draw_text(image_draw, text, (box[0], box[1]-10), color="white")
    return np.asarray(image_drawn)
