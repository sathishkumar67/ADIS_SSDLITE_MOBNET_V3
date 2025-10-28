"""Model definition for SSDLite with MobileNetV3 backbone.

This module wraps torchvision's prebuilt SSDLite+MobileNetV3 model and
exposes a small convenience class (`SSDLITE_MOBILENET_V3_Large`) that adjusts
the classification head to match the number of classes in our dataset and
provides an optimizer helper.
"""

from __future__ import annotations
from typing import Tuple
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils


class SSDLITE_MOBILENET_V3_Large(nn.Module):
    def __init__(self, num_classes_with_bg:int, img_size: int=320) -> None:
        """
        Initialize the SSD MobileNet V3 Large model.

        Args:
            num_classes_with_bg (int): The number of object classes in the dataset, including the background class.
            img_size (int): The input image size for the model. Default is 320.
            
        Returns:
            None
        """
        super().__init__()
        
        # initialize attributes
        self.num_classes_with_bg = num_classes_with_bg
        self.img_size = img_size
        
        # initialize the model
        self.model = ssdlite320_mobilenet_v3_large(weights='COCO_V1', weights_backbone="DEFAULT")
        
        # modify the model to use the specified number of classes
        self.model.head.classification_head = SSDLiteClassificationHead(
            in_channels=det_utils.retrieve_out_channels(self.model.backbone, (self.img_size, self.img_size)),
            num_anchors=self.model.anchor_generator.num_anchors_per_location(),
            num_classes=self.num_classes_with_bg,
            norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        )
        # set the nominal number of detections per image
        self.model.detections_per_img = 100 

    
    def configure_optimizers(self, lr: float = 0.0001, betas: Tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.0001, eps: float = 1e-08, fused: bool = True) -> torch.optim.Optimizer:  
        """
        Configure the optimizer for the model.

        Args:
            lr (float): The learning rate for the optimizer. Default is 0.0001.
            betas (Tuple[float, float]): The beta values for the Adam optimizer. Default is (0.9, 0.999).
            weight_decay (float): The weight decay for the optimizer. Default is 0.0001.
            eps (float): The epsilon value for the optimizer. Default is 1e-08.
            fused (bool): Whether to use a fused version of Adam. Default is True.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """      
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        # Create AdamW optimizer and use the fused version if available 
        return optim.AdamW([{'params': decay_params, 'weight_decay': weight_decay},
                                    {'params': nodecay_params, 'weight_decay': 0.0}], 
                                    lr=lr, 
                                    betas=betas, 
                                    eps=eps, 
                                    fused=fused)
    
    def forward(self, images: torch.Tensor, targets: dict=None):
        """
        Forward pass through the model.

        Args:
            images (torch.Tensor): Input images for the model.
            targets (dict): Target annotations for the images. Default is None.

        Returns:
            torch.Tensor: Model output predictions.
        """
        return self.model(images, targets)