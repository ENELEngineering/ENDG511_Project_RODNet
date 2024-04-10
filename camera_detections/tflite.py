from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from parameters import Parameters

from utils import (
    validate_path, 
    resize, 
    normalize,
    yolo2xyxy, 
    batch_iou
)
from timeit import timeit
import numpy as np
import torchvision
import torch

class TFliteRunner:
    """
    Runs TensorFlow Lite models.
    
    Parameters
    ----------
        model: str or tflite interpreter.
            The path to the model or the loaded tflite model.

        parameters: Parameters
            These are the model parameters set from the command line.

        labels: list
            Unique string labels.
    """
    def __init__(
        self,
        model,
        parameters: Parameters,
        labels: list=None
    ):
        try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
            from tflite_runtime.interpreter import ( 
                Interpreter, 
                load_delegate
            ) 
        except ImportError:
            import tensorflow as tf
            Interpreter, load_delegate = ( # NOSONAR
                tf.lite.Interpreter,
                tf.lite.experimental.load_delegate,
            )

        if isinstance(model, str):
            model = validate_path(model)
            self.interpreter = Interpreter(model_path=model)  # load TFLite model
        else:
            self.interpreter = model
            self.model = "Training Model"

        self.interpreter.allocate_tensors()  # allocate
        self.parameters = parameters
        
        self.labels = []
        if labels is not None:
            self.labels = labels
    
        if self.parameters.warmup > 0:
            t = timeit(self.interpreter.invoke, number=self.parameters.warmup)

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method performs detections given a numpy image.

        Parameters
        ----------
            image: np.ndarray
                This is a numpy array image.

        Returns
        -------
            boxes: np.ndarray
                These are the non-normalized bounding boxes of the 
                detections in the format [[xmin, ymin, xmax, ymax], [...], ...].

            scores: np.ndarray
                These are the scores of each bounding box.

            labels: np.ndarray
                These are the labels of each bounding box.
        """
        height, width, _ = image.shape
        boxes, labels, scores = self.run_single_instance(image)
        
        # Filter bounding boxes based on the score threshold.
        indices = scores > self.parameters.acceptance_score
        boxes = boxes[indices, ...]
        scores = scores[indices]
        labels = labels[indices]

        # Scale boxes to be non-normalized to the image dimensions.
        boxes[..., 0] = boxes[..., 0] * width
        boxes[..., 1] = boxes[..., 1] * height
        boxes[..., 2] = boxes[..., 2] * width
        boxes[..., 3] = boxes[..., 3] * height
        boxes = boxes.astype(np.int16)

        return boxes, scores, labels
              
    def run_single_instance(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce tflite predictions on one image. 
        This method does not pad the images to match the input shape of the 
        model. This is different to yolov5 implementation where images are 
        padded: https://github.com/ultralytics/yolov5/blob/master/val.py#L193

        Tflite runner functionality was taken from:: \
        https://github.com/ultralytics/yolov5/blob/master/models/common.py#L601

        Parameters
        ----------
            image: np.ndarray
                The numpy image to process.

        Returns
        -------
            nmsed_boxes: np.ndarray
                The prediction bounding boxes.. [[box1], [box2], ...].

            nmsed_classes: np.ndarray
                The prediction labels.. [cl1, cl2, ...]. A label of 0
                represents barcodes and a label of 1 means QR codes.

            nmsed_scores: np.ndarray
                The prediction confidence scores.. [score, score, ...]
                normalized between 0 and 1.
        """

        """Input Preprocessing"""
        # Take only the (height, width).
        height, width = self.get_input_shape()[1:3]
        tensor = resize(image, (width, height))
        tensor = normalize(tensor, self.parameters.normalization, self.get_input_type())
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], tensor)
        
        """Inference"""
        self.interpreter.invoke()
        y = []
        for output in output_details:
            # is TFLite quantized uint8 model.
            int8 = input_details[0]["dtype"] == np.uint8  
            x = self.interpreter.get_tensor(output["index"])
            if int8:
                scale, zero_point = output["quantization"]
                x = (x.astype(np.float32) - zero_point) * scale  # re-scale
            y.append(x)
        y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
        h, w = self.get_input_shape()[1:3]
        # NMS requires non-normalized coordinates.
        y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels.
        if isinstance(y, (list, tuple)):
            output_data = self.from_numpy(y[0]) if len(y) == 1 else [
                self.from_numpy(x) for x in y]
        else:
            output_data = self.from_numpy(y)

        """Postprocessing"""
        output = self.non_max_supression(output_data)
        nmsed_boxes, nmsed_classes, nmsed_scores = self.postprocessing(output)
        return nmsed_boxes, nmsed_classes, nmsed_scores

    def from_numpy(self, x: np.ndarray):
        """
        Convert numpy array to torch tensor.

        Parameters
        ----------
            x: np.ndarray
                The numpy array to convert to pytorch tensor.

        Returns
        -------
            x: torch.Tensor
                This is the numpy array as a torch.Tensor type.
        """
        return torch.from_numpy(x).to("cpu") if isinstance(x, np.ndarray) else x
    
    def non_max_supression( # NOSONAR
            self, 
            prediction,
            agnostic: bool=False,
            multi_label: bool=True,
            nm: int=0,
            max_wh: int=7680,
            max_nms: int= 30000,
            redundant: bool=True,
            merge: bool=False
        ):
        """
        This is the YoloV5 NMS found here:: \
        https://github.com/ultralytics/yolov5/blob/master/utils/general.py#L955

        Reproducing the same parameters as YoloV5 requires:: \
        1) detection score threshold = 0.001
        2) detection iou threshold = 0.60
        3) max detections = 300

        Parameters
        ----------
            prediction: torch.Tensor
                Raw predictions from the model (inference_out, loss_out).
            agnostic: bool

            multi_label: bool
                If validation has more than 1 labels.

            nm: int

            max_wh: int
                The maximum box width and height (pixels).

            max_nms: int
                The maximum number of boxes into torchvision.ops.nms().

            redundant: bool
                Require redundant detections.

            merge: bool
                Use merge NMS.

        Returns
        -------
            output
        """
        # YOLOv5 model in validation model.
        if isinstance(prediction, (list, tuple)):  
            # Select only inference output.
            prediction = prediction[0] 

        bs = prediction.shape[0]  # Batch size.
        nc = prediction.shape[2] - nm - 5  # The number of classes.
        xc = prediction[..., 4] > self.parameters.detection_score # Candidates.

        multi_label &= nc > 1  # Multiple labels per box (adds 0.5ms/img).
        mi = 5 + nc  # Mask start index.
        output = [torch.zeros((0, 6 + nm), device="cpu")] * bs

        for xi, x in enumerate(prediction):  # Image index, image inference.
            x = x[xc[xi]]  # Confidence.
            if not x.shape[0]:
                continue
            
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf.
            
            # (center_x, center_y, width, height) to (x1, y1, x2, y2).
            box = yolo2xyxy(x[:, :4])  
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls).
            if multi_label:
                i, j = (
                    x[:, 5:mi] > self.parameters.detection_score
                ).nonzero(as_tuple=False).T
                x = torch.cat(
                    (box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 
                    1)
            else:  # Best class only.
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat(
                    (box, conf, j.float(), mask), 
                    1)[conf.view(-1) > self.parameters.detection_score]

            # Check shape.
            n = x.shape[0]  # Number of boxes.
            if not n:  # No boxes.
                continue
            # Sort by confidence and remove excess boxes.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  

            # Batched NMS.
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # The classes.
            # boxes (offset by class), scores.
            boxes, scores = x[:, :4] + c, x[:, 4]  
            
            # Torchvision NMS.
            i = torchvision.ops.nms(
                boxes, scores, self.parameters.detection_iou) 
            i = i[:self.parameters.max_detections]  
            # Merge NMS (boxes merged using weighted mean).
            if merge and (1 < n < 3e3):  
                # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4).
                # IoU matrix.
                iou = batch_iou(boxes[i], boxes) > self.parameters.detection_iou 
                weights = iou * scores[None]  # Box weights.
                # Merged boxes.
                x[i, :4] = torch.mm(
                    weights, x[:, :4]).float() / weights.sum(1, keepdim=True) 
                if redundant:
                    i = i[iou.sum(1) > 1]  # Require redundancy.
            output[xi] = x[i]
        return output

    def postprocessing(
        self, 
        output: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves the boxes, scores and labels.

        Parameters
        ----------
            outputs:
                This contains bounding boxes, scores, labels in the format.
                [[xmin, ymin, xmax, ymax, confidence, label], [...], ...].

        Returns
        -------
            nmsed_boxes: np.ndarray
                The prediction bounding boxes.. [[box1], [box2], ...].

            nmsed_classes: np.ndarray
                The prediction labels.. [cl1, cl2, ...].

            nmsed_scores: np.ndarray
                The prediction confidence scores.. [score, score, ...]
                normalized between 0 and 1.
        """
        h, w = self.get_input_shape()[1:3]
        outputs = output[0].numpy()
        indices = np.nonzero(outputs[..., 4:5]>=self.parameters.acceptance_score)
        outputs = outputs[indices[0], ...]
        outputs[..., :4] /= [w, h, w, h]
        
        nmsed_boxes = outputs[..., :4]
        # Single dimensional arrays gets converted to the element. 
        # Specify the axis into 1 to prevent that.
        nmsed_scores = np.squeeze(outputs[..., 4:5], axis=1)
        nmsed_classes = np.squeeze(outputs[...,5:6], axis=1)
        if len(self.labels) > 0:
            string_nms_predicted_classes = list()
            for cls in nmsed_classes:
                string_nms_predicted_classes.append(self.labels[int(cls)])
            nmsed_classes = np.array(string_nms_predicted_classes)
            
        return nmsed_boxes, nmsed_classes, nmsed_scores
    
    def get_input_type(self) -> str:
        """
        This returns the input type of the model.

        Returns
        -------
            type: str
                The input type of the model.
        """
        return self.interpreter.get_input_details()[0]["dtype"].__name__
    
    def get_output_type(self) -> str:
        """
        This returns the output type of the model.

        Returns
        -------
            type: str
                The output type of the model.
        """
        return self.interpreter.get_output_details()[0]["dtype"].__name__

    def get_input_shape(self) -> np.ndarray:
        """
        Grabs the model input shape.

        Returns
        -------
            shape: np.ndarray
                The model input shape.
                (batch size, height, width, channels).
        """
        return self.interpreter.get_input_details()[0]["shape"]
    
    def get_output_shape(self) -> np.ndarray:
        """
        Grabs the model output shape.

        Returns
        --------
            shape: np.ndarray
                The model output shape.
                (batch size, boxes, classes).
        """
        return self.interpreter.get_output_details()[0]["shape"]