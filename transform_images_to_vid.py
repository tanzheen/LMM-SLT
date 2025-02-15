import cv2
import os
import glob
from pathlib import Path
import re

def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def images_to_video(image_folder, output_path, fps=25):
    """
    Convert a sequence of images in a folder to a video file.
    
    Args:
        image_folder (str): Path to the folder containing images
        output_path (str): Path where the video will be saved
        fps (int): Frames per second for the output video
    """
    # Get all image files and sort them naturally
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return False
    
    # Sort files naturally (1.png, 2.png, ..., 10.png)
    image_files.sort(key=natural_sort_key)
    
    # Read first image to get dimensions
    frame = cv2.imread(image_files[0])
    if frame is None:
        print(f"Failed to read first image in {image_folder}")
        return False
    
    height, width = frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Failed to read image: {image_file}")
    
    out.release()
    return True

def process_folders(base_path):
    """
    Process all folders in the base path and create videos from image sequences.
    
    Args:
        base_path (str): Path containing folders with image sequences
    """
    base_path = Path(base_path)
    for phase in ["train", "dev","test"]:
    # Get all subdirectories
        base_path_phase = os.path.join(base_path, phase)
        folders = [os.path.join(base_path_phase, f) for f in os.listdir( base_path_phase) if os.path.isdir(os.path.join(base_path_phase, f))]
    
        for folder in folders:
            print(f"Processing folder: {os.path.basename(folder)}")
            output_video = os.path.join(base_path_phase, f"{os.path.basename(folder)}.mp4")
            
            success = images_to_video(str(folder), str(output_video), fps=25)
            if success:
                print(f"Created video: {output_video}")
            else:
                print(f"Failed to create video for folder: {folder}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert image sequences to videos")
    parser.add_argument("base_path", default = "../PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/", help="Path containing folders with image sequences")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second for the output videos")
    
    args = parser.parse_args()
    process_folders(args.base_path)
