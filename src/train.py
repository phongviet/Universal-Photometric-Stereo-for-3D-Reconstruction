from UniPS import *
from data.PSWildDataset import *
import time
import re
import yaml


def load_config(config_path='configs.yaml'):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to YAML configuration file (default: 'configs.yaml')

    Returns:
        dict: Configuration dictionary containing all parameters:
            - paths: Dataset and output directories
            - training: Training hyperparameters
            - testing: Evaluation settings
            - model: Model architecture configuration
            - resume: Checkpoint resume settings
            - device: Computing device (cuda/cpu)

    Example:
        >>> config = load_config('configs.yaml')
        >>> batch_size = config['training']['batch_size']
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config_path='configs.yaml'):
    """
    Main training function for Universal Photometric Stereo.

    Training Pipeline:
        1. Load configuration from YAML file
        2. Initialize model and optimizer
        3. Check for existing checkpoints and resume if available
        4. Load training dataset with multi-view images
        5. Training loop:
           - For each batch:
             a) Forward pass through encoder, aggregation, prediction
             b) Compute loss: L = (1/S) Σ ||n̂_p - n_p^gt||₂²
             c) Backward pass and optimizer step
           - Save checkpoint after each epoch
        6. Automatic resumption from last checkpoint if interrupted

    Mathematical Formulation:
        Given batch of B objects, each with N images {I_k}_{k=1}^N:

        Loss per object:
            L_obj = (1/S) Σ_{p∈S} ||Normalize(f_θ(I_1,...,I_N)_p) - n_p^gt||₂²

        where:
        - S: Random sample of num_samples valid pixels
        - f_θ: Neural network with parameters θ
        - n^gt: Ground truth normal map

        Total batch loss:
            L_batch = (1/B) Σ_{i=1}^B L_obj^i

        Update rule (AdamW):
            θ ← θ - α · AdamW(∇_θ L_batch)

    Args:
        config_path (str): Path to configuration file (default: 'configs.yaml')

    Configuration Parameters Used:
        - paths.train_root: Training data directory
        - paths.output_dir: Output/checkpoint directory
        - training.batch_size: Batch size
        - training.learning_rate: Learning rate
        - training.epochs: Number of training epochs
        - training.img_size: Image resolution
        - resume.enabled: Enable automatic resume
        - device: GPU/CPU selection

    Output:
        Saves checkpoints after each epoch:
        - encoder_ep{N}.pth: Encoder weights
        - agg_ep{N}.pth: Aggregation module weights
        - pred_ep{N}.pth: Prediction head weights

    Example:
        >>> train_model('configs.yaml')
        Using device: cuda
        Found checkpoint for Epoch 2. Loading weights...
        Starting Training loop from Epoch 3 to 4...
    """
    # Load configuration
    config = load_config(config_path)

    # Extract parameters
    TRAIN_ROOT = config['paths']['train_root']
    OUTPUT_DIR = config['paths']['output_dir']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = config['training']['batch_size']
    LR = config['training']['learning_rate']
    EPOCHS = config['training']['epochs']
    IMG_SIZE = config['training']['img_size']
    MAX_IMAGES = config['training']['max_images']
    LIGHTING_TYPE = config['training']['lighting_type']
    NUM_WORKERS = config['training']['num_workers']

    device_name = config['device']
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Model ---
    unips = UniPSNet(device, lr=LR)

    # --- Resume Logic ---
    start_epoch = 0
    resume_enabled = config['resume']['enabled']
    load_epoch = config['resume']['load_epoch']

    if resume_enabled:
        # Look for encoder checkpoints in output dir
        existing_ckpts = glob.glob(os.path.join(OUTPUT_DIR, "encoder_ep*.pth"))

        if len(existing_ckpts) > 0:
            # Extract epoch numbers
            epochs_found = []
            for ckpt_path in existing_ckpts:
                match = re.search(r'encoder_ep(\d+).pth', ckpt_path)
                if match:
                    epochs_found.append(int(match.group(1)))

            if len(epochs_found) > 0:
                # Use specified epoch or latest
                if load_epoch is not None and load_epoch in epochs_found:
                    latest_epoch = load_epoch
                else:
                    latest_epoch = max(epochs_found)

                print(f"Found checkpoint for Epoch {latest_epoch}. Loading weights...")

                try:
                    unips.encoder.load_state_dict(torch.load(f"{OUTPUT_DIR}/encoder_ep{latest_epoch}.pth", map_location=device))
                    unips.aggregation.load_state_dict(torch.load(f"{OUTPUT_DIR}/agg_ep{latest_epoch}.pth", map_location=device))
                    unips.prediction.load_state_dict(torch.load(f"{OUTPUT_DIR}/pred_ep{latest_epoch}.pth", map_location=device))

                    start_epoch = latest_epoch + 1
                    print(f"Successfully loaded. Resuming training from Epoch {start_epoch}.")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}. Starting from scratch.")
                    start_epoch = 0
        else:
            print("No checkpoints found. Starting training from scratch.")
    else:
        print("Resume disabled. Starting training from scratch.")

    # --- Initialize Data Loaders ---
    if os.path.exists(TRAIN_ROOT):
        train_dataset = PSWildDataset(TRAIN_ROOT, mode='Train', max_images=MAX_IMAGES,
                                       img_size=IMG_SIZE, lighting_type=LIGHTING_TYPE)
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=NUM_WORKERS)
        print(f"Training dataset loaded: {len(train_dataset)} objects")
    else:
        print(f"Warning: Training root {TRAIN_ROOT} not found!")
        train_loader = None

    # --- Training Loop ---
    if train_loader:
        if start_epoch < EPOCHS:
            print(f"Starting Training loop from Epoch {start_epoch} to {EPOCHS - 1}...")
            for epoch in range(start_epoch, EPOCHS):
                unips.train()
                total_loss = 0
                start_time = time.time()

                for batch_idx, batch in enumerate(train_loader):
                    try:
                        loss = unips.step(batch, decoder_imgsize=(IMG_SIZE, IMG_SIZE),
                                          encoder_imgsize=(256, 256), mode='Train')
                        total_loss += loss

                        if batch_idx % 10 == 0:
                            print(f"Epoch [{epoch}/{EPOCHS - 1}] Batch {batch_idx} Loss: {loss:.4f}")
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        continue

                if len(train_loader) > 0:
                    avg_loss = total_loss / len(train_loader)
                    print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}. Time: {time.time() - start_time:.1f}s")

                # Save Checkpoint
                torch.save(unips.encoder.state_dict(), f"{OUTPUT_DIR}/encoder_ep{epoch}.pth")
                torch.save(unips.aggregation.state_dict(), f"{OUTPUT_DIR}/agg_ep{epoch}.pth")
                torch.save(unips.prediction.state_dict(), f"{OUTPUT_DIR}/pred_ep{epoch}.pth")
                print(f"Checkpoint saved for epoch {epoch}")
        else:
            print(f"Training already completed up to epoch {start_epoch - 1} (Target: {EPOCHS}).")
    else:
        print("No training data available!")

    print("Training completed!")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs.yaml'
    train_model(config_path)
