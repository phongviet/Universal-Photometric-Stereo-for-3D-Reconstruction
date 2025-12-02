import os
import glob
import cv2
import numpy as np
import torch
import torch.utils.data as data


class PSWildDataset(data.Dataset):
    """
    Dataset loader for PS-Wild (Photometric Stereo in the Wild) dataset.

    Loads multi-view images of objects under varying lighting conditions along with
    ground truth surface normal maps. Supports different lighting types and handles
    various image formats and bit depths.

    Data Structure:
        root_dir/
        ├── object1.data/
        │   ├── 00000.tif, 00001.tif, ...  (input images)
        │   ├── normal.tif                  (ground truth normal map)
        │   └── ...
        └── object2.data/
            └── ...

    Preprocessing Pipeline:
        1. Load N images: {I_k}_{k=1}^N
        2. Resize to target size: I_k → I_k^{H×W}
        3. Normalize to [0, 1]: I = I / max_value
        4. Load normal map: n^gt ∈ [0, 1]³ → n^gt ∈ [-1, 1]³
        5. Create mask: M = {p | ||n_p^gt||₂ > 0.1}
        6. Normalize images by mean: I_k(p) = I_k(p) / mean(I_k)

    Mathematical Formulation:
        For each valid pixel p:
        - Input: {I_k(p)}_{k=1}^N where I_k(p) ∈ ℝ³
        - Output: n_p ∈ ℝ³ with ||n_p||₂ = 1
        - Mask: M(p) ∈ {0, 1}

    Args:
        root_dir (str): Root directory containing .data folders
        mode (str): 'Train' or 'Test' - affects image sampling strategy
        max_images (int): Maximum number of images to load per object
        img_size (int): Target image size (images are resized to img_size × img_size)
        lighting_type (str): Type of lighting conditions:
            - 'Directional': Point light sources
            - 'Environment': Environment map lighting
            - 'DirEnv': Mix of directional and environment
            - 'all': All available images (for training)
    """

    def __init__(self, root_dir, mode='Train', max_images=32, img_size=256, lighting_type='all'):
        """
        Initialize PS-Wild dataset.

        Args:
            root_dir (str): Root directory containing .data folders
            mode (str): 'Train' or 'Test' (default: 'Train')
            max_images (int): Maximum images per object (default: 32)
            img_size (int): Target image size in pixels (default: 256)
            lighting_type (str): Lighting type filter (default: 'all')
        """
        self.mode = mode
        self.max_images = max_images
        self.img_size = img_size
        self.lighting_type = lighting_type

        # Recursive search for .data folders
        # Your structure indicates folders like "accessory.obj_..." inside PSWildTest
        self.objlist = sorted(glob.glob(os.path.join(root_dir, '*', '*.data')))
        if len(self.objlist) == 0:
            # Try flat structure just in case
            self.objlist = sorted(glob.glob(os.path.join(root_dir, '*.data')))

        print(f"Found {len(self.objlist)} objects in {root_dir} with mode {mode}")

    def __len__(self):
        """
        Get the number of objects in the dataset.

        Returns:
            int: Number of objects
        """
        return len(self.objlist)

    def __getitem__(self, index):
        """
        Load and preprocess one object's data.

        Processing steps:
            1. Load N images under different lighting conditions
            2. Resize all images to (img_size, img_size)
            3. Normalize pixel values to [0, 1]
            4. Load ground truth normal map and convert to [-1, 1]
            5. Create validity mask: M(p) = 1 if ||n_p||₂ > 0.1
            6. Normalize each image by its mean intensity over valid pixels:
               I_k^norm(p) = I_k(p) / mean({I_k(q) | M(q) = 1})

        Args:
            index (int): Index of the object to load

        Returns:
            tuple: (images, normals, mask) where:
                - images: torch.Tensor of shape [C, H, W, N]
                    RGB images with C=3 channels, N views
                - normals: torch.Tensor of shape [C, H, W]
                    Ground truth normal map, C=3 (x, y, z components)
                    Normalized to [-1, 1] range
                - mask: torch.Tensor of shape [1, H, W]
                    Binary mask indicating valid pixels (1) vs background (0)

        Note:
            - Images are mean-normalized per view
            - Normal vectors satisfy: n ∈ [-1, 1]³, ||n||₂ ≈ 1 for valid pixels
            - All tensors are float32
        """
        obj_path = self.objlist[index]

        # 1. Identify valid image files based on lighting_type
        if self.lighting_type == 'Directional':
            pattern = 'Directional_*.tif'
        elif self.lighting_type == 'Environment':
            pattern = 'Environment_*.tif'
        elif self.lighting_type == 'DirEnv':
            pattern = 'DirEnv_*.tif'
        else:  # 'all' or Default Training behavior (00000.tif etc)
            pattern = '*.tif'

        img_files = sorted(glob.glob(os.path.join(obj_path, pattern)))

        # Filter out Ground Truth and Aux files if 'all' pattern caught them
        exclude_files = ['baseColor.tif', 'depth.exr', 'metal.tif', 'normal.tif', 'roughness.tif']
        img_files = [f for f in img_files if os.path.basename(f) not in exclude_files]

        if len(img_files) == 0:
            # Fallback for training sets that might just use numbers (00000.tif)
            img_files = sorted(glob.glob(os.path.join(obj_path, '[0-9]*.tif')))

        if len(img_files) == 0:
            raise RuntimeError(f"No valid images found in {obj_path} for type {self.lighting_type}")

        # 2. Select Subset
        total_imgs = len(img_files)
        if self.mode == 'Train':
            indices = np.random.permutation(total_imgs)[:self.max_images]
        else:
            # For testing, we usually take the first N or a fixed stride
            indices = range(min(total_imgs, self.max_images))

        images = []
        for idx in indices:
            path = img_files[idx]
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: continue

            # Handle standard RGB conversion
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            # Normalize (16bit or 8bit)
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32) / 255.0
            images.append(img)

        if len(images) == 0:
            raise RuntimeError(f"Could not load images from {obj_path}")

        images = np.stack(images, axis=3)  # H, W, 3, N

        # 3. Load Normal (Ground Truth)
        # You mentioned 'normal.tif' is explicitly present in the folder
        normal_path = os.path.join(obj_path, 'normal.tif')

        if os.path.exists(normal_path):
            nml = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            if nml is None:
                # Fallback if read fails
                nml = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            else:
                nml = cv2.cvtColor(nml, cv2.COLOR_BGR2RGB)
                nml = cv2.resize(nml, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                if nml.dtype == np.uint16:
                    nml = nml.astype(np.float32) / 65535.0
                else:
                    nml = nml.astype(np.float32) / 255.0
                nml = 2.0 * nml - 1.0  # [0, 1] -> [-1, 1]

            # Create valid mask
            mask = np.linalg.norm(nml, axis=2) > 0.1
            mask = mask.astype(np.float32)
        else:
            nml = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            mask = np.ones((self.img_size, self.img_size), dtype=np.float32)

        # 4. Normalization (Divide by mean)
        for i in range(images.shape[3]):
            valid_pixels = images[:, :, :, i][mask > 0]
            if valid_pixels.size > 0:
                mean_val = np.mean(valid_pixels)
                if mean_val > 1e-6:
                    images[:, :, :, i] /= mean_val

        # 5. Format for PyTorch
        # Output: Images [C, H, W, N], Normal [C, H, W], Mask [1, H, W]
        images = images.transpose(2, 0, 1, 3)
        nml = nml.transpose(2, 0, 1)
        mask = mask[np.newaxis, :, :]

        return torch.from_numpy(images), torch.from_numpy(nml), torch.from_numpy(mask)