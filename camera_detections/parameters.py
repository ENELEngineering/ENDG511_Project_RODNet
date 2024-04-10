class Parameters:
    """
    This class defines the parameters to use for the object
    detection models.

    Parameters
    ----------
        detection_score: float
            This is the score threshold to set for the NMS.

        detection_iou: float
            This is the IoU threshold to set for the NMS.
        
        acceptance_score: float
            This is score threshold to accept as valid detections.

        max_detections: int
            This is the maximum number of detections for the NMS.

        normalization: str
            This is the type of image normalization to perform before
            passing the image to the model.

        warmup: int
            This is the number of warmup iterations to perform for the
            model prior to deploying it for inference.
    """
    def __init__(
        self,
        detection_score: float = 0.001,
        detection_iou: float = 0.60,
        acceptance_score: float = 0.25,
        max_detections: int = 300,
        normalization: str = "unsigned",
        warmup: int = 0,
        sharpen: int = 0
    ) -> None:
        
        self._detection_score = detection_score
        self._detection_iou = detection_iou
        self._acceptance_score = acceptance_score
        self._max_detections = max_detections
        self._normalization = normalization
        self._warmup = warmup

    @property
    def detection_score(self) -> float:
        """
        Access the detection score property.

        Returns
        -------
            detection_score: float
                This is the score threshold to pass to the NMS.
        """
        return self._detection_score
    
    @detection_score.setter
    def detection_score(self, score: float):
        """
        Set a new value to the detection score.

        Parameters
        ----------
            score: float
                This is the new score threshold to set.
        """
        self._detection_score = score

    @property
    def detection_iou(self) -> float:
        """
        Access the detection IoU property.

        Returns
        -------
            detection_iou: float
                This is the IoU threshold to pass to the NMS.
        """
        return self._detection_iou
    
    @detection_iou.setter
    def detection_iou(self, iou: float):
        """
        Set a new value to the detection IoU.

        Parameters
        ----------
            score: float
                This is the new IoU threshold to set.
        """
        self._detection_iou = iou

    @property
    def acceptance_score(self) -> float:
        """
        Access the acceptance score property.

        Returns
        -------
            acceptance_score: float
                This is the score threshold to consider as valid detections.
        """
        return self._acceptance_score
    
    @acceptance_score.setter
    def acceptance_score(self, score: float):
        """
        Set a new value to the acceptance score.

        Parameters
        ----------
            score: float
                This is the new score threshold to set.
        """
        self._acceptance_score = score

    @property
    def max_detections(self) -> int:
        """
        Access the maximum detections property.

        Returns
        -------
            max_detections: int
                This is the maximum detections to set for the NMS.
        """
        return self._max_detections
    
    @max_detections.setter
    def max_detections(self, num: int):
        """
        Set a new value to the maximum detections parameter.

        Parameters
        ----------
            num: int
                This new value to set for the maximum detections.
        """
        self._max_detections = num

    @property
    def normalization(self) -> str:
        """
        Access the normalization property.

        Returns
        -------
            normalization: str
                This is the image normalization to perform. Currently 
                supports signed, unsigned, and raw normalizations.
        """
        return self._normalization
    
    @normalization.setter
    def normalization(self, norm: str):
        """
        Set a new value to the normalization type.

        Parameters
        ----------
            norm: str
                This is the new normalization type to set.
        """
        self._normalization = norm

    @property
    def warmup(self) -> int:
        """
        Access the warmup property.

        Returns
        -------
            warmup: int
                This is the warmup to perform for the model prior to 
                inference.
        """
        return self._warmup
    
    @warmup.setter
    def warmup(self, num: int):
        """
        Set a new value to the warmup parameter.

        Parameters
        ----------
            num: int
                This new value to set for the warmup.
        """
        self._warmup = num