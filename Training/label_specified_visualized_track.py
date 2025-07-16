import os
import torch
import face_alignment
import numpy as np
import cv2
from tqdm import tqdm

# Configuration
input_dir = 'ps/scratch/face2d3d/train'
landmark_output_dir = 'ps/scratch/face2d3d/train_annotated_torch7'
image_output_dir = 'ps/scratch/face2d3d/train_annotated_images'
min_size = 256
train_file = 'ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_train_list_max_normal_100_ring_5_1_serial.npy'
val_file = 'ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_val_list_max_normal_100_ring_5_1_serial.npy'

# Create output directories
os.makedirs(landmark_output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

# Initialize face alignment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device.upper()}")
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device=device
)

def draw_landmarks(image, landmarks):
    """Draw facial landmarks on an image"""
    # Convert to BGR for OpenCV drawing
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw each landmark point
    for point in landmarks:
        x, y = point
        cv2.circle(vis_image, (int(x), int(y)), 2, (0, 255, 0), -1)  # Green filled circle
    
    # Draw facial feature connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # Jawline
        (17, 18), (18, 19), (19, 20), (20, 21),  # Right eyebrow
        (22, 23), (23, 24), (24, 25), (25, 26),  # Left eyebrow
        (27, 28), (28, 29), (29, 30),  # Nose bridge
        (31, 32), (32, 33), (33, 34), (34, 35),  # Lower nose
        (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # Right eye
        (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # Left eye
        (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), 
        (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # Outer lips
        (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 60)   # Inner lips
    ]
    
    for start, end in connections:
        x1, y1 = landmarks[start]
        x2, y2 = landmarks[end]
        cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
    
    return vis_image

def process_images_from_datafile(data_file, set_name):
    """Process all images listed in a data file"""
    try:
        data = np.load(data_file, allow_pickle=True)
        image_paths = data.ravel()
        
        print(f"\nProcessing {len(image_paths)} {set_name} images")
        
        processed_count = 0
        skipped_count = 0
        existing_count = 0
        no_landmarks_count = 0
        
        # Create progress bar
        pbar = tqdm(total=len(image_paths), desc=f"{set_name.capitalize()} Images")
        
        for rel_path in image_paths:
            img_path = os.path.join(input_dir, rel_path + '.jpg')
            landmark_path = os.path.join(landmark_output_dir, rel_path + '.npy')
            vis_path = os.path.join(image_output_dir, rel_path + '.jpg')
            
            # Update progress bar
            pbar.update(1)
            
            # Skip if both outputs exist
            if os.path.exists(landmark_path) and os.path.exists(vis_path):
                existing_count += 1
                pbar.set_postfix_str(f"✅: {processed_count}, ⏩: {existing_count}, ❌: {skipped_count} ({no_landmarks_count} no LM)")
                continue
                
            # Skip if image doesn't exist
            if not os.path.exists(img_path):
                skipped_count += 1
                pbar.set_postfix_str(f"✅: {processed_count}, ⏩: {existing_count}, ❌: {skipped_count} ({no_landmarks_count} no LM)")
                continue
                
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    skipped_count += 1
                    pbar.set_postfix_str(f"✅: {processed_count}, ⏩: {existing_count}, ❌: {skipped_count} ({no_landmarks_count} no LM)")
                    continue
                    
                # Resize if too small
                h, w = image.shape[:2]
                scale = 1.0
                if h < min_size or w < min_size:
                    scale = max(min_size / h, min_size / w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                    
                # Convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Predict landmarks
                preds = fa.get_landmarks(image_rgb)
                if not preds:
                    no_landmarks_count += 1
                    skipped_count += 1
                    pbar.set_postfix_str(f"✅: {processed_count}, ⏩: {existing_count}, ❌: {skipped_count} ({no_landmarks_count} no LM)")
                    continue
                    
                landmarks = preds[0]
                
                # Create output directories
                os.makedirs(os.path.dirname(landmark_path), exist_ok=True)
                os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                
                # Save landmarks
                np.save(landmark_path, landmarks)
                
                # Generate and save visualized image
                vis_image = draw_landmarks(image_rgb, landmarks)
                cv2.imwrite(vis_path, vis_image)
                
                processed_count += 1
                pbar.set_postfix_str(f"✅: {processed_count}, ⏩: {existing_count}, ❌: {skipped_count} ({no_landmarks_count} no LM)")
                
            except Exception as e:
                skipped_count += 1
                pbar.set_postfix_str(f"✅: {processed_count}, ⏩: {existing_count}, ❌: {skipped_count} ({no_landmarks_count} no LM)")
                continue
                
        pbar.close()
        print(f"Completed {set_name} set: "
              f"{processed_count} processed, "
              f"{existing_count} already existed, "
              f"{skipped_count} skipped "
              f"({no_landmarks_count} with no landmarks)")
        return processed_count
        
    except Exception as e:
        print(f"\nError processing {set_name} set: {str(e)}")
        return 0

# Process training and validation images
print("="*50)
train_count = process_images_from_datafile(train_file, "training")

print("\n" + "="*50)
val_count = process_images_from_datafile(val_file, "validation")

print("\n" + "="*50)
print("PROCESSING COMPLETE")
print("="*50)
print(f"Total training images processed: {train_count}")
print(f"Total validation images processed: {val_count}")
print(f"Total images processed in this run: {train_count + val_count}")
print(f"\nLandmarks saved to: {landmark_output_dir}")
print(f"Annotated images saved to: {image_output_dir}")