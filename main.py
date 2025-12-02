#!/usr/bin/env python
"""
Universal Photometric Stereo - Main Entry Point

Usage:
    python main.py train [--config CONFIG_PATH]
    python main.py eval [--config CONFIG_PATH] [--epoch EPOCH]
    python main.py inference INPUT_FOLDER [--output OUTPUT_PATH] [--config CONFIG_PATH] [--epoch EPOCH]

Examples:
    python main.py train
    python main.py eval --epoch 4
    python main.py inference ./test_images --output ./result.png
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))


def main():
    """
    Main entry point for Universal Photometric Stereo system.

    Provides a unified command-line interface for three main operations:
    1. Training: Train the neural network on photometric stereo data
    2. Evaluation: Evaluate model performance on test set
    3. Inference: Predict normal maps from new image sets

    Command Structure:
        python main.py <command> [options]

    Available Commands:
        train      - Train the model using configuration from configs.yaml
        eval       - Evaluate trained model on test set
        inference  - Run inference on a folder of images

    Common Options:
        --config CONFIG_PATH  : Path to YAML configuration file
        --epoch EPOCH        : Specific checkpoint epoch to use
        --output OUTPUT_PATH : Custom output path (inference only)

    Examples:
        # Training with default config
        >>> python main.py train

        # Evaluation with specific checkpoint
        >>> python main.py eval --epoch 4

        # Inference on custom data
        >>> python main.py inference ./my_images --output ./result.png

        # Using custom configuration
        >>> python main.py train --config custom_config.yaml

    Returns:
        None (exits with code 0 on success, 1 on error)

    See Also:
        - configs.yaml: Configuration file with all parameters
        - README.md: Detailed documentation
        - QUICKSTART.md: Quick start guide
    """
    parser = argparse.ArgumentParser(
        description='Universal Photometric Stereo - Train, Evaluate, or Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='configs.yaml',
                            help='Path to configuration file (default: configs.yaml)')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--config', type=str, default='configs.yaml',
                           help='Path to configuration file (default: configs.yaml)')
    eval_parser.add_argument('--epoch', type=int, default=None,
                           help='Specific epoch to evaluate (default: latest)')

    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on a folder of images')
    inference_parser.add_argument('input_folder', type=str,
                                help='Path to folder containing input images')
    inference_parser.add_argument('--output', type=str, default=None,
                                help='Path to save output normal map (default: auto-generated)')
    inference_parser.add_argument('--config', type=str, default='configs.yaml',
                                help='Path to configuration file (default: configs.yaml)')
    inference_parser.add_argument('--epoch', type=int, default=None,
                                help='Specific epoch to use (default: latest)')

    args = parser.parse_args()

    # Check if command is provided
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == 'train':
        from src.train import train_model
        print("=" * 60)
        print("TRAINING MODE")
        print("=" * 60)
        train_model(args.config)

    elif args.command == 'eval':
        from src.eval import evaluate_model
        print("=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        evaluate_model(args.config, args.epoch)

    elif args.command == 'inference':
        from src.inference import run_inference
        print("=" * 60)
        print("INFERENCE MODE")
        print("=" * 60)
        if not os.path.exists(args.input_folder):
            print(f"Error: Input folder '{args.input_folder}' does not exist!")
            sys.exit(1)
        run_inference(args.input_folder, args.output, args.config, args.epoch)


if __name__ == "__main__":
    main()

