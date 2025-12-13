import os
import cv2
import argparse
import re
import sys

def sort_epochs(folder_names):
    # Extract numbers and sort
    def extract_number(name):
        match = re.search(r'epoch_(\d+)', name)
        return int(match.group(1)) if match else -1
    
    return sorted([f for f in folder_names if extract_number(f) != -1], key=extract_number)

def create_video(samples_dir, output_dir, fps=5):
    # Find all epoch folders
    try:
        epoch_folders = [d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d)) and 'epoch_' in d]
    except FileNotFoundError:
        print(f"Error: Directory '{samples_dir}' not found.")
        return

    sorted_epochs = sort_epochs(epoch_folders)
    
    if not sorted_epochs:
        print("No epoch folders found.")
        return

    print(f"Found {len(sorted_epochs)} epoch folders.")

    # Find all sample images in the last epoch to know what to look for
    # We check the last one because sometimes early epochs might fail or have different files
    last_epoch_path = os.path.join(samples_dir, sorted_epochs[-1])
    sample_files = [f for f in os.listdir(last_epoch_path) if f.endswith('.png')]
    
    if not sample_files:
        print("No images found in the last epoch folder.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for sample_file in sample_files:
        print(f"Processing {sample_file}...")
        
        # Determine video properties from first image
        first_img_path = None
        for ep in sorted_epochs:
             path = os.path.join(samples_dir, ep, sample_file)
             if os.path.exists(path):
                 first_img_path = path
                 break
        
        if not first_img_path:
            print(f"Could not find any images for {sample_file}")
            continue
        
        frame = cv2.imread(first_img_path)
        if frame is None:
            print(f"Could not read image {first_img_path}")
            continue
            
        height, width, layers = frame.shape
        video_name = os.path.join(output_dir, f"evolution_{os.path.splitext(sample_file)[0]}.mp4")
        
        # Define codec
        # mp4v is widely supported
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        frames_written = 0
        for epoch_folder in sorted_epochs:
            img_path = os.path.join(samples_dir, epoch_folder, sample_file)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue

                # Resize if dimensions changed (unlikely but safe)
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))

                # Add Epoch Text
                epoch_match = re.search(r'epoch_(\d+)', epoch_folder)
                epoch_num = epoch_match.group(1) if epoch_match else "?"
                text = f"Epoch: {epoch_num}"
                
                # Text settings
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                color = (255, 255, 255) # White
                thickness = 2
                
                # Get text size to position it
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = 20
                text_y = 40
                
                # Add black background for text for better visibility
                # (x1, y1), (x2, y2)
                cv2.rectangle(img, (text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
                cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
                
                video.write(img)
                frames_written += 1
            else:
                # Optional: hold previous frame? For now, just skip.
                pass

        video.release()
        print(f"Saved {video_name} ({frames_written} frames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from training epochs")
    parser.add_argument("--samples_dir", type=str, required=True, help="Path to the samples directory containing epoch_XXX folders")
    parser.add_argument("--output_dir", type=str, default="videos", help="Path to save generated videos")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    
    args = parser.parse_args()
    create_video(args.samples_dir, args.output_dir, args.fps)
