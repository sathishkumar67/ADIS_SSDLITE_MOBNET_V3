from __future__ import annotations
from typing import Tuple
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset



class SSDLITEOBJDET_DATASET(Dataset):
    def __init__(self, root_dir: str, split: str, num_classes: int, img_size: int=320, mode: str="train", dtype=np.float32) -> None:
        """
        Initialize the SSDLITEOBJDET dataset.

        Args:
            root_dir (str): The root directory of the dataset.
            split (str): The split of the dataset, either 'train' or 'eval'.
            num_classes (int): The number of object classes in the dataset.
            img_size (int, optional): The size of the input images. Defaults to 320.
            mode (str, optional): The mode of the dataset, either 'train' or 'eval'. Defaults to 'train'.
            dtype (type, optional): The data type for the bounding box coordinates. Defaults to np.float32.
        
        Returns:
            None
        """
        super().__init__()
        
        # initialize attributes
        self.root_dir, self.split, self.img_size, self.num_classes = root_dir, split.lower(), img_size, num_classes
        self.current_dir = os.path.join(self.root_dir, self.split)
        self.mode, self.dtype =  mode, dtype

        # check if model is train or eval mode
        if self.mode not in ["train", "eval"]:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'train' or 'eval'.")
        
        # set interpolation method for resizing
        # LANCZOS4 is a high-quality resampling filter, suitable for training.
        # INTER_LINEAR is a bilinear interpolation method, suitable for evaluation.
        self.interpolation = cv2.INTER_LANCZOS4 if self.mode == "train" else cv2.INTER_LINEAR

        # Validate current directory exists and is a directory
        if not os.path.exists(self.current_dir):
            raise FileNotFoundError(f"{self.current_dir} does not exist.")
        elif not os.path.isdir(self.current_dir):
            raise NotADirectoryError(f"{self.current_dir} is not a directory.")
        
        # check if the split directory is empty
        if len(os.listdir(self.current_dir)) == 0:
            raise ValueError(f"The directory {self.current_dir} is empty.")
        
        # get image and label files
        self.image_files = sorted(
            [os.path.join(self.current_dir, f) for f in os.listdir(self.current_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: os.path.splitext(x)[0]
        )
        self.label_files = [os.path.join(self.current_dir, os.path.splitext(f)[0] + '.txt') for f in self.image_files]

        # Validate existence for ALL label files
        for img_file, lbl_file in zip(self.image_files, self.label_files):
            if not os.path.exists(lbl_file):
                raise FileNotFoundError(f"Label file missing for {img_file}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        img_path, label_path = self.image_files[idx], self.label_files[idx]

        # Read image and convert to RGB format
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        orig_height, orig_width, _ = image.shape
        # tensor with uint8 datatype 
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=self.interpolation)
        
        # Read label file and parse the bounding boxes and labels
        data = np.loadtxt(label_path, dtype=self.dtype, delimiter=' ', ndmin=2)
        
        if data.size == 0:
            return image, {
                'boxes': np.array([[0.0, 0.0, 1.0, 1.0]], dtype=self.dtype),
                'labels': np.array(0, dtype=np.uint8)
            }
        else:
            # Convert normalized box coordinates into absolute coordinates, where orig_width and orig_height are your original dimensions.
            cx, cy, w, h = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
            xmin = np.maximum(0, (cx - w/2) * orig_width)
            ymin = np.maximum(0, (cy - h/2) * orig_height)
            xmax = np.minimum(orig_width, (cx + w/2) * orig_width)
            ymax = np.minimum(orig_height, (cy + h/2) * orig_height)
            
            # Filter degenerate boxes (width or height less than 1)
            valid_mask = ((xmax - xmin) >= 1) & ((ymax - ymin) >= 1)
            valid_boxes = np.stack([xmin[valid_mask], ymin[valid_mask],
                                    xmax[valid_mask], ymax[valid_mask]], axis=1)

            # Adjust class IDs (cid from first column)
            valid_labels = data[valid_mask, 0].astype(np.uint8) 
            np.add(valid_labels, 1, out=valid_labels)  # Increment class IDs by 1 for background class

            # scale boxes to new image size
            scale_factors = np.array([self.img_size / orig_width, self.img_size / orig_height,
                                    self.img_size / orig_width, self.img_size / orig_height], dtype=valid_boxes.dtype)
            np.multiply(valid_boxes, scale_factors, out=valid_boxes)
            
            # Validate class IDs
            if np.all((valid_labels < 0) & (valid_labels >= self.num_classes)):
                raise ValueError(f"Invalid class ID in {label_path}")

            if self.mode == "train":
                np.divide(valid_boxes, self.img_size, out=valid_boxes) # Normalize boxes to [0, 1]

            return image, {
                'boxes': valid_boxes,
                'labels': valid_labels}
        
    def denormalize_bbox(self, boxes: torch.Tensor|np.ndarray) -> torch.Tensor|np.ndarray:
        # Denormalize boxes to original size
        return boxes * self.img_size
    
    def normalize_bbox(self, boxes: torch.Tensor|np.ndarray) -> torch.Tensor|np.ndarray:
        # Normalize boxes to [0, 1]
        return boxes / self.img_size
    
    def denormalize_image(self, image: torch.Tensor|np.ndarray) -> torch.Tensor|np.ndarray:
        # Denormalize image to [0, 255]
        return image * 255.0
    
    def normalize_image(self, image: torch.Tensor|np.ndarray) -> torch.Tensor|np.ndarray:
        # Normalize image to [0, 1]
        return image / 255.0
    

def collate_fn(batch):
    """
    Collate function to process a batch of samples from SSDLITEOBJDET_DATASET.
    
    Args:
        batch: List of tuples containing (image, target_dict)
    
    Returns:
        Tuple of (images, targets) where:
        - images: Tensor of shape (B, C, H, W) with normalized images
        - targets: List of dicts with 'boxes' and 'labels' tensors for each image
    """
    images = []
    targets = []

    # Process each sample in the batch
    for img, tgt in batch:
        # Convert HWC numpy array to CHW tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        img_tensor /= 255.0
        images.append(img_tensor)
        
        # Convert annotations to tensors
        boxes = torch.as_tensor(tgt['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(tgt['labels'], dtype=torch.int64)
        
        targets.append({
            'boxes': boxes,
            'labels': labels
        })
    
    return torch.stack(images, dim=0), targets