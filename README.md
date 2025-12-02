# Universal Photometric Stereo

A deep learning system for photometric stereo - estimating surface normals from multiple images under different lighting conditions.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.7+
- CUDA

## Configuration

All parameters are centralized in `configs.yaml`. Edit this file to customize:

- **Dataset paths**: Training and testing data locations
- **Training parameters**: Batch size, learning rate, epochs, etc.
- **Testing parameters**: Evaluation settings
- **Inference parameters**: Settings for running inference on new data
- **Model parameters**: Architecture configuration

### Example Configuration

```yaml
paths:
  train_root: "./data/train"
  test_root: "./data/test"
  output_dir: "./output"
  checkpoint_dir: "./pretrained_weights"

training:
  batch_size: 2
  learning_rate: 0.0001
  epochs: 5
  img_size: 256
  max_images: 10

testing:
  lighting_type: 'Directional'  # Options: 'Directional', 'Environment', 'DirEnv', 'all'
  max_images: 32

device: "cuda"  # or "cpu"
```

## Usage

### 1. Training

Train the model using the default configuration:

```bash
python main.py train
```

Train with a custom configuration file:

```bash
python main.py train --config my_config.yaml
```


### 2. Evaluation

Evaluate the model on the test set using the latest checkpoint:

```bash
python main.py eval
```

Evaluate a specific epoch:

```bash
python main.py eval --epoch 4
```

Evaluate with custom configuration:

```bash
python main.py eval --config my_config.yaml --epoch 4
```

**Outputs:**
- Mean Angular Error (MAE) for each test object
- Final average MAE across all objects
- Visualization images saved to `./output/results_inference/`

### 3. Inference

Run inference on a folder of images to generate a normal map:

```bash
python main.py inference ./path/to/images
```

Specify output path:

```bash
python main.py inference ./path/to/images --output ./result_normal.png
```

Use a specific checkpoint:

```bash
python main.py inference ./path/to/images --epoch 4
```

**Input Format:**
- Place all images of the same object under different lighting conditions in a single folder
- Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP
- The system will automatically process and normalize the images

**Output:**
- A normal map saved as PNG (RGB image where RGB channels represent XYZ normal directions)
- Default output location: `./output/inference_results/<folder_name>_normal.png`

## Data Format

### Training/Testing Data Structure

```
data_root/
├── object1.data/
│   ├── 00000.tif          # Input images under different lighting
│   ├── 00001.tif
│   ├── ...
│   ├── normal.tif         # Ground truth normal map
│   └── ...
├── object2.data/
│   └── ...
```

### Lighting Types

The dataset supports different lighting conditions:
- **Directional**: Files named `Directional_*.tif`
- **Environment**: Files named `Environment_*.tif`
- **DirEnv**: Files named `DirEnv_*.tif`
- **all**: All available images (for training)

## Model Checkpoints

Checkpoints are automatically saved after each training epoch:

```
output/
├── encoder_ep0.pth
├── agg_ep0.pth
├── pred_ep0.pth
├── encoder_ep1.pth
├── ...
```

### Loading Checkpoints

- **Training**: Automatically loads the latest checkpoint if available
- **Evaluation**: Use `--epoch` flag or automatically loads latest
- **Inference**: Use `--epoch` flag or automatically loads latest

## Advanced Usage

### Custom Configuration

Create a custom `configs.yaml` file with your specific parameters:

```yaml
# Custom configuration
paths:
  train_root: "/my/custom/path"
  output_dir: "./my_output"

training:
  batch_size: 4
  learning_rate: 0.0002
  epochs: 10
  
testing:
  lighting_type: 'Environment'
```

Then use it:

```bash
python main.py train --config custom_configs.yaml
```


## Citation


## License



