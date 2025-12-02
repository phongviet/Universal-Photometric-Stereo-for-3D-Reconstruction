import torch
import gc
import os
import cv2
import numpy as np
import glob
import re
import yaml
from UniPS import UniPSNet
from data.PSWildDataset import PSWildDataset
import torch.utils.data as data
from utils import angular_error


def load_config(config_path='configs.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(config_path='configs.yaml', checkpoint_epoch=None):
    """Main evaluation function"""
    config = load_config(config_path)

    # Extract parameters
    TEST_ROOT = config['paths']['test_root']
    CHECKPOINT_DIR = config['paths'].get('checkpoint_dir', config['paths']['output_dir'])
    OUTPUT_DIR = os.path.join(config['paths']['output_dir'], 'results_inference')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TEST_LIGHTING = config['testing']['lighting_type']
    IMG_SIZE = config['testing']['img_size']
    MAX_IMAGES = config['testing']['max_images']
    BATCH_SIZE = config['testing']['batch_size']
    NUM_WORKERS = config['testing']['num_workers']

    device_name = config['device']
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize model
    unips = UniPSNet(device, lr=0.0)

    # Find checkpoint
    if checkpoint_epoch is None:
        encoder_files = glob.glob(os.path.join(CHECKPOINT_DIR, "**/encoder_ep*.pth"), recursive=True)
        if len(encoder_files) == 0:
            raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")

        epochs = []
        for f in encoder_files:
            match = re.search(r'encoder_ep(\d+)\.pth', f)
            if match:
                epochs.append(int(match.group(1)))
        checkpoint_epoch = max(epochs)

        for f in encoder_files:
            if f"encoder_ep{checkpoint_epoch}.pth" in f:
                CHECKPOINT_DIR = os.path.dirname(f)
                break

    # Load weights
    encoder_path = os.path.join(CHECKPOINT_DIR, f"encoder_ep{checkpoint_epoch}.pth")
    agg_path = os.path.join(CHECKPOINT_DIR, f"agg_ep{checkpoint_epoch}.pth")
    pred_path = os.path.join(CHECKPOINT_DIR, f"pred_ep{checkpoint_epoch}.pth")

    if os.path.exists(encoder_path):
        print(f"Loading weights from Epoch {checkpoint_epoch}...")
        unips.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        unips.aggregation.load_state_dict(torch.load(agg_path, map_location=device))
        unips.prediction.load_state_dict(torch.load(pred_path, map_location=device))
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Could not find checkpoints at {encoder_path}")

    unips.eval()

    # Prepare test dataset
    if os.path.exists(TEST_ROOT):
        test_dataset = PSWildDataset(TEST_ROOT, mode='Test', max_images=MAX_IMAGES,
                                       img_size=IMG_SIZE, lighting_type=TEST_LIGHTING)
        test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=NUM_WORKERS)
        print(f"Test Set Loaded: {len(test_dataset)} objects.")
    else:
        print(f"Test root {TEST_ROOT} not found.")
        return

    # Inference loop
    print(f"Starting Inference on '{TEST_LIGHTING}' set...")
    total_mae = 0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            _, gt_nml, mask = batch
            gt_nml = gt_nml.to(device)
            mask = mask.to(device)
            valid_mask = mask > 0.5

            try:
                pred_nml = unips.step(batch, decoder_imgsize=(IMG_SIZE, IMG_SIZE),
                                       encoder_imgsize=(256, 256), mode='Test')

                mae = angular_error(pred_nml, gt_nml, valid_mask).item()
                total_mae += mae
                count += 1
                print(f"Object {i}: MAE = {mae:.2f}")

                # Save visualization
                pred_np = pred_nml[0].permute(1, 2, 0).cpu().numpy()
                pred_np = (pred_np + 1) / 2.0
                pred_np = np.clip(pred_np * 255, 0, 255).astype(np.uint8)
                pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{OUTPUT_DIR}/res_{i}.png", pred_bgr)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Object {i} skipped due to CUDA OOM.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print(f"Error on object {i}: {e}")

    if count > 0:
        print(f"\nFinal Average MAE ({TEST_LIGHTING}): {total_mae / count:.4f}")
    else:
        print("No valid objects processed.")

    print("Evaluation completed!")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs.yaml'
    checkpoint_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else None
    evaluate_model(config_path, checkpoint_epoch)

