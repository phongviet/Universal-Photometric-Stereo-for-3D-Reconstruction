import torch
import torch.nn.functional as F
import os
import glob
import cv2
import numpy as np
import yaml
from UniPS import UniPSNet


def load_config(config_path='configs.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_images_from_folder(folder_path, max_images=32, img_size=256):
    """
    Load images from a folder for inference
    Returns: images tensor [C, H, W, N], mask [1, H, W]
    """
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
    img_files = []
    for ext in image_extensions:
        img_files.extend(glob.glob(os.path.join(folder_path, ext)))

    img_files = sorted(img_files)

    if len(img_files) == 0:
        raise RuntimeError(f"No images found in {folder_path}")

    # Limit to max_images
    img_files = img_files[:max_images]
    print(f"Found {len(img_files)} images in {folder_path}")

    images = []
    for img_path in img_files:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping...")
            continue

        # Handle grayscale and color images
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # Normalize
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        images.append(img)

    if len(images) == 0:
        raise RuntimeError(f"Could not load any valid images from {folder_path}")

    # Stack images: H, W, 3, N
    images = np.stack(images, axis=3)

    # Create a simple mask (all ones for inference)
    mask = np.ones((img_size, img_size), dtype=np.float32)

    # Normalize each image by its mean
    for i in range(images.shape[3]):
        mean_val = np.mean(images[:, :, :, i])
        if mean_val > 1e-6:
            images[:, :, :, i] /= mean_val

    # Convert to PyTorch format: [C, H, W, N] and [1, H, W]
    images = images.transpose(2, 0, 1, 3)  # C, H, W, N
    mask = mask[np.newaxis, :, :]  # 1, H, W

    return torch.from_numpy(images), torch.from_numpy(mask)


def run_inference(input_folder, output_path=None, config_path='configs.yaml', checkpoint_epoch=None):
    """
    Run inference on a folder of images to estimate surface normal map.

    This function takes multiple images of the same object under different lighting
    conditions and predicts a surface normal map without requiring ground truth.

    Inference Pipeline:
        1. Load N images from input folder: {I_k}_{k=1}^N
        2. Preprocess images (resize, normalize, mean-norm)
        3. Load pre-trained model weights from checkpoint
        4. Forward pass through network:
           a) Encoder: Extract multi-view features
           b) Aggregation: Combine information across views
           c) Prediction: Regress to surface normals
        5. Post-process and save normal map as RGB image

    Mathematical Formulation:
        Given N input images {I_k}_{k=1}^N:

        1. Feature extraction:
           F_k = Encoder(I_1, ..., I_N)_k ∈ ℝ^{256×H/4×W/4}

        2. For each pixel p:
           - Sample features: f_k(p) for k ∈ {1,...,N}
           - Aggregate: h_p = Transformer([I_1(p), f_1(p)], ..., [I_N(p), f_N(p)])
           - Predict: n̂_p = PredictionHead(h_p)
           - Normalize: n̂_p = n̂_p / ||n̂_p||₂

        3. Output normal map:
           N̂ ∈ ℝ^{H×W×3} where ||n̂_p||₂ = 1 for all pixels p

        4. Convert to image:
           N_img = (N̂ + 1) / 2 × 255 ∈ [0, 255]³

    Args:
        input_folder (str): Path to folder containing input images
        output_path (str, optional): Path to save output normal map.
                                    If None, auto-generates path in config output_dir.
        config_path (str): Path to configuration file (default: 'configs.yaml')
        checkpoint_epoch (int, optional): Specific epoch to load.
                                         If None, uses latest checkpoint.

    Returns:
        str: Path where the normal map was saved

    Output:
        Saves normal map as PNG image where:
        - RGB channels represent (X, Y, Z) components of surface normal
        - Values in [0, 255] map to normal components in [-1, 1]
        - Conversion: normal_component = (pixel_value / 255) * 2 - 1

    Raises:
        FileNotFoundError: If no checkpoints found or input folder doesn't exist
        RuntimeError: If no valid images found in input folder

    Example:
        >>> output = run_inference('./my_object_images/', checkpoint_epoch=4)
        Loading images from ./my_object_images/...
        Found 12 images in ./my_object_images/
        Loading checkpoint from epoch 4...
        Running inference...
        Normal map saved to: ./output/inference_results/my_object_images_normal.png

    Note:
        - Input images should be of the same object under different lighting
        - More images generally produce better results (recommended: 10-30)
        - No ground truth normal map required
        - Processing is done in chunks to avoid out-of-memory errors
    """
    # Load configuration
    config = load_config(config_path)

    # Setup device
    device_name = config['device']
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Get parameters
    img_size = config['inference']['img_size']
    max_images = config['inference']['max_images']
    chunk_size = config['inference']['chunk_size']
    checkpoint_dir = config['paths']['checkpoint_dir']

    # Load images
    print(f"Loading images from {input_folder}...")
    images, mask = load_images_from_folder(input_folder, max_images, img_size)

    # Initialize model
    print("Initializing model...")
    unips = UniPSNet(device, lr=0.0)

    # Load checkpoint
    if checkpoint_epoch is None:
        # Find latest checkpoint
        encoder_files = glob.glob(os.path.join(checkpoint_dir, "**/encoder_ep*.pth"), recursive=True)
        if len(encoder_files) == 0:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        # Extract epoch numbers
        import re
        epochs = []
        for f in encoder_files:
            match = re.search(r'encoder_ep(\d+)\.pth', f)
            if match:
                epochs.append(int(match.group(1)))
        checkpoint_epoch = max(epochs)

        # Find the directory containing this epoch
        for f in encoder_files:
            if f"encoder_ep{checkpoint_epoch}.pth" in f:
                checkpoint_dir = os.path.dirname(f)
                break

    # Load weights
    encoder_path = os.path.join(checkpoint_dir, f"encoder_ep{checkpoint_epoch}.pth")
    agg_path = os.path.join(checkpoint_dir, f"agg_ep{checkpoint_epoch}.pth")
    pred_path = os.path.join(checkpoint_dir, f"pred_ep{checkpoint_epoch}.pth")

    print(f"Loading checkpoint from epoch {checkpoint_epoch}...")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Checkpoint not found: {encoder_path}")

    unips.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    unips.aggregation.load_state_dict(torch.load(agg_path, map_location=device))
    unips.prediction.load_state_dict(torch.load(pred_path, map_location=device))
    print("Weights loaded successfully.")

    # Set to evaluation mode
    unips.eval()

    # Prepare batch
    images = images.unsqueeze(0)  # Add batch dimension: [1, C, H, W, N]
    mask = mask.unsqueeze(0)  # [1, 1, H, W]

    # Create dummy normal map (not used in inference)
    nml = torch.zeros(1, 3, img_size, img_size)

    batch = (images, nml, mask)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        pred_nml = unips.step(
            batch,
            decoder_imgsize=(img_size, img_size),
            encoder_imgsize=(256, 256),
            mode='Test'
        )

    # Convert to numpy and save
    pred_np = pred_nml[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

    # Denormalize from [-1, 1] to [0, 1]
    pred_np = (pred_np + 1) / 2.0
    pred_np = np.clip(pred_np * 255, 0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

    # Determine output path
    if output_path is None:
        output_dir = config['inference']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        folder_name = os.path.basename(os.path.normpath(input_folder))
        output_path = os.path.join(output_dir, f"{folder_name}_normal.png")

    # Save result
    cv2.imwrite(output_path, pred_bgr)
    print(f"Normal map saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <input_folder> [output_path] [checkpoint_epoch]")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    checkpoint_epoch = int(sys.argv[3]) if len(sys.argv) > 3 else None

    run_inference(input_folder, output_path, checkpoint_epoch=checkpoint_epoch)

