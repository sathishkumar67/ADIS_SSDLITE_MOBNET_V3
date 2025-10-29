from typing import Union, List, Tuple
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def draw_detections(
    image_or_path: Union[str, Image.Image, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    classes: List[str],
    conf_thresh: float = 0.2,
    input_size: int = 320,
    show: bool = True
) -> np.ndarray:
    """
    Run `model` on a single image and draw detection boxes + labels.

    Args:
        image_or_path: Path to an image, a PIL Image, or a torch Tensor image
                       shaped (1,C,H,W) or (C,H,W) with pixel values in [0,1].
        model: Torch detection model (returns list[dict] like torchvision detectors).
        device: torch.device where model & tensors should live (e.g., `cuda`).
        classes: List of class names (background excluded). Labels in outputs
                 are expected to be 1-indexed (1..N).
        conf_thresh: Score threshold for displaying detections (default 0.2).
        input_size: Size for resizing when loading image_path/PIL.Image (default 320).
        show: If True, display the annotated image with matplotlib.

    Returns:
        Annotated image as a NumPy array (H, W, 3), dtype=uint8 (the image shown).
    """
    # Prepare image tensor (batch dimension expected)
    if isinstance(image_or_path, str):
        pil = Image.open(image_or_path).convert("RGB")
        preprocess = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])
        img_tensor = preprocess(pil).unsqueeze(0)  # 1, C, H, W
        print(f"[draw_detections] Loaded image from path: {image_or_path}")
    elif isinstance(image_or_path, Image.Image):
        pil = image_or_path.convert("RGB")
        preprocess = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])
        img_tensor = preprocess(pil).unsqueeze(0)
        print("[draw_detections] Using provided PIL.Image")
    elif isinstance(image_or_path, torch.Tensor):
        t = image_or_path.detach().cpu()
        if t.ndim == 3:
            img_tensor = t.unsqueeze(0)  # add batch
        elif t.ndim == 4:
            img_tensor = t
        else:
            raise ValueError("Tensor image must be 3D (C,H,W) or 4D (B,C,H,W)")
        # If values are not float, convert and scale to [0,1] if necessary
        if img_tensor.dtype != torch.float32:
            img_tensor = img_tensor.float()
        # assume already in [0,1]
        print("[draw_detections] Using provided torch.Tensor image")
    else:
        raise TypeError("image_or_path must be a path, PIL.Image, or torch.Tensor")

    # Move to device
    img_tensor = img_tensor.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        print("[draw_detections] Running model inference...")
        outputs = model(img_tensor)

    if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
        raise RuntimeError("Model returned no outputs")

    out = outputs[0]
    boxes = out.get("boxes", torch.empty((0, 4))).cpu()
    scores = out.get("scores", torch.empty((0,))).cpu()
    labels = out.get("labels", torch.empty((0,), dtype=torch.long)).cpu()

    # Filter by confidence
    keep_mask = scores > conf_thresh
    boxes = boxes[keep_mask].numpy() if keep_mask.any() else np.zeros((0, 4))
    scores = scores[keep_mask].numpy() if keep_mask.any() else np.zeros((0,))
    labels = labels[keep_mask].numpy() if keep_mask.any() else np.zeros((0,), dtype=int)

    print(f"[draw_detections] {len(boxes)} detections (score > {conf_thresh:.2f})")

    # Prepare an HWC uint8 image for plotting (use the model input resolution)
    img_disp = img_tensor[0].cpu().permute(1, 2, 0).numpy()  # H,W,C float [0,1]
    img_disp = np.clip(img_disp * 255.0, 0, 255).astype(np.uint8)

    # Plot
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_disp)
    ax.axis("off")

    for box, score, lbl in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1

        # Rectangle and label
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Resolve class name (labels are assumed 1-indexed)
        try:
            class_name = classes[int(lbl) - 1]
        except Exception:
            class_name = str(int(lbl))

        caption = f"{class_name}: {score:.2f}"
        ax.text(x1, y1 - 6, caption, fontsize=10, color='white',
                bbox=dict(facecolor='red', alpha=0.7, pad=1, edgecolor='none'))

    plt.tight_layout()
    if show:
        plt.show()

    return img_disp