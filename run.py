#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone script to run VGGT model inference.
Usage: python vggt/run.py --input_dir <path> --output_dir <path>
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch

# Add the directory containing this script to the path
# This allows importing from the vggt package subdirectory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def load_model(device="cuda"):
    """Load and initialize the VGGT model."""
    print("Initializing and loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print("Model loaded successfully.")
    return model


def run_inference(input_dir, output_dir, model=None, device="cuda"):
    """
    Run the VGGT model on images in the input_dir and save predictions to output_dir.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save predictions
        model: Pre-loaded model (if None, will load it)
        device: Device to run inference on

    Returns:
        dict: Predictions dictionary
    """
    print(f"Processing images from {input_dir}")

    # Device check
    if not torch.cuda.is_available() and device == "cuda":
        print("WARNING: CUDA is not available. Falling back to CPU.")
        device = "cpu"

    # Load model if not provided
    if model is None:
        model = load_model(device)
    else:
        model = model.to(device)
        model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(input_dir, "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")

    if len(image_names) == 0:
        raise ValueError(f"No images found in {input_dir}")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None  # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    print("Saving images data...")
    images_np = images.cpu().numpy()
    print(f"DEBUG: images shape before squeeze: {images_np.shape}")
    # Remove batch dimension if it exists and equals 1
    if images_np.shape[0] == 1:
        images_np = images_np.squeeze(0)
    predictions["images"] = images_np

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "predictions.npz")
    print(f"Saving predictions to {output_path}")
    np.savez(output_path, **predictions)

    # Clean up
    torch.cuda.empty_cache()
    print("Inference complete!")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run VGGT model inference on images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    run_inference(args.input_dir, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()

