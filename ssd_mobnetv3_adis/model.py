from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import time
from tqdm import tqdm
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
from torchmetrics.detection import MeanAveragePrecision 



@dataclass  
class SSD_MOBILENET_V3_Large_Config:
    """
    Configuration class for SSD MobileNet V3 Large model in PyTorch Lightning.
    """
    classes: List[str]
    num_classes_with_bg: int 
    img_size: int = 320
    lr: float = 0.0001
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0001
    eps: float = 1e-08
    fused: bool = True



class SSD_MOBILENET_V3_Large(nn.Module):
    def __init__(self, num_classes_with_bg:int, img_size: int=320) -> None:
        """
        Initialize the SSD MobileNet V3 Large model.

        Args:
            num_classes_with_bg (int): The number of object classes in the dataset, including the background class.
            img_size (int): The input image size for the model. Default is 320.
            
        Returns:
            None
        """
        super(SSD_MOBILENET_V3_Large, self).__init__()
        
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
            norm_layer=partial(nn.BatchNorm2d, eps=1e-8, momentum=0.03)
        )
        # self.model.detections_per_img = 100 # need to test this
    
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
    
    def forward(self, images: torch.Tensor, targets: dict=None) :
        """
        Forward pass through the model.

        Args:
            images (torch.Tensor): Input images for the model.
            targets (dict): Target annotations for the images. Default is None.

        Returns:
            torch.Tensor: Model output predictions.
        """
        return self.model(images, targets)
    
    def load(self, checkpoint_path: str, key_name: str = "model_state_dict", map_location: str = "cpu") -> None:
        """
        Load the model state dict from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            key_name (str): Key name in the checkpoint file to load the model state dict.
            map_location (str): Map location for loading the checkpoint.
        """
        start_time = time.time()
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.load_state_dict(torch.load(checkpoint_path, map_location=map_location)[key_name])
        print(f"Checkpoint loaded in {time.time() - start_time:.2f} seconds.")
        
    def evaluate(self, dataloaders: dict[str, torch.utils.data.DataLoader], device: torch.device|str) -> dict[str, dict[str, float]]:
        """
        Evaluate the model on the given dataloaders.
        
        Args:
            dataloaders (dict[str, torch.utils.data.DataLoader]): A dictionary containing the dataloaders for each split (train, val, test).
            device (torch.device|str): The device to run the evaluation on.
        """
        
        # initialize the metric
        metric = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        extended_summary=True)
        
        # get the dataloaders
        splits = dataloaders.keys()
        loaders = [dataloaders[split] for split in splits]
        
        # if device is a string
        if isinstance(device, str):
            device = device
        else:
            device = f"{device.type}:{device.index}"
        
        # run the evaluation
        print("Starting evaluation...")
        results = {}
        for split, loader in zip(splits, loaders):
            print(f"Evaluating {split} set")
            # set the model to evaluation mode
            self.eval()
            # reset the metric 
            metric.reset()
            progress_bar = tqdm(loader, desc=f"Evaluating {split} set", unit="batch")
            with torch.no_grad():
                for images, targets in progress_bar:
                    images = torch.as_tensor(images, dtype=torch.float32, device=device)
                    images.div_(255.0)

                    for target in targets:
                        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32, device=device)
                        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64, device=device)
                    
                    # pass the images only to the model
                    outputs = self(images)
                    # update the metric with the outputs and targets
                    metric.update(outputs, targets)
                # compute the metric, store the results
                results[split] =  metric.compute()
                
        print("Evaluation complete.")
        return results