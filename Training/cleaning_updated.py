import numpy as np
import os
import cv2
import torch
from tqdm import tqdm
import face_alignment
from facenet_pytorch import MTCNN
from sklearn.model_selection import train_test_split
import random

class VGGFace2DataCleaner:
    def __init__(self, root_folder, output_dir, use_landmarks_for_bbx=True, val_size=0.2, random_seed=42):
        """
        Args:
            root_folder: Path to VGGFace2 dataset root (contains identity folders)
            output_dir: Directory to save output files
            use_landmarks_for_bbx: If True, uses landmarks to define face region
            val_size: Fraction of identities to use for validation
            random_seed: Seed for reproducible train/val split
        """
        self.root_folder = root_folder
        self.output_dir = output_dir
        self.use_landmarks_for_bbx = use_landmarks_for_bbx
        self.val_size = val_size
        self.random_seed = random_seed
        
        # Initialize face alignment
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector='blazeface',
            flip_input=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        if not use_landmarks_for_bbx:
            self.mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # File paths
        self.train_file = os.path.join(output_dir, 'vggface2_train_list_max_normal_100_ring_5_1_serial.npy')
        self.val_file = os.path.join(output_dir, 'vggface2_val_list_max_normal_100_ring_5_1_serial.npy')
        self.state_file = os.path.join(output_dir, 'processing_state.npz')
        
        # Initialize state
        self.processed_identities = set()
        self.train_data = np.empty((0, 5), dtype=object)
        self.val_data = np.empty((0, 5), dtype=object)
        self.all_identities = []
        self.train_identities = set()
        self.val_identities = set()
        
        # Load existing state if available
        self.load_state()

# [Keep all helper methods unchanged: get_face_region, expand_bbx, etc.]
    def get_face_region(self, image):
        """Get face bounding box using face-alignment's detector with proper resizing"""
        try:
            # Convert to RGB if needed
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # BlazeFace works best with 128px width while maintaining aspect ratio
            target_width = 128
            height, width = image.shape[:2]
            scale = target_width / float(width)
            target_height = int(height * scale)
            
            # Resize image for BlazeFace
            resized = cv2.resize(image, (target_width, target_height))
            
            # Detect faces
            detections = self.fa.face_detector.detect_from_image(resized)
            if len(detections) == 0:
                return None
                
            # Take the first face and scale coordinates back to original size
            x1, y1, x2, y2, _ = detections[0]
            return np.array([
                x1 / scale,
                y1 / scale,
                x2 / scale,
                y2 / scale
            ])
        except Exception as e:
            print(f"Face detection failed: {str(e)}")
            return None

    def expand_bbx(self, bbx, expand_top=0.1, expand_sides=0.2):
        """Expand bounding box by given percentages."""
        x1, y1, x2, y2 = bbx
        bw, bh = x2 - x1, y2 - y1
        return np.array([
            x1 - expand_sides * bw,
            y1 - expand_top * bh,
            x2 + expand_sides * bw,
            y2 + expand_sides * bh
        ])

    def shift_bbx(self, bbx, shift_ratio=0.05):
        """Shift bbx by 5% to bottom-right."""
        bw, bh = bbx[2] - bbx[0], bbx[3] - bbx[1]
        ε = np.array([bw, bh]) * shift_ratio
        return bbx + np.array([ε[0], ε[1], ε[0], ε[1]])

    def check_landmark_consistency(self, k1, k2, bbx):
        """Check if max normalized landmark difference < 0.1."""
        bw, bh = bbx[2] - bbx[0], bbx[3] - bbx[1]
        D = np.diag([1/bw, 1/bh])
        ε = np.array([bw/20, bh/20])
        diff = k2 - ε - k1
        max_diff = np.max(np.linalg.norm(np.dot(D, diff.T).T, axis=1))
        return max_diff < 0.1

    def process_image(self, img_path):
        """Process a single image through the cleaning pipeline."""
        image = cv2.imread(img_path)
        if image is None:
            return None
        
        # Skip small images entirely
        if min(image.shape[:2]) < 64:  # 64px minimum
            return None
                
        bbx = self.get_face_region(image)
        if bbx is None:
            return None
            
        original_expanded = self.expand_bbx(bbx)
        shifted_expanded = self.expand_bbx(self.shift_bbx(bbx))
        
        k1 = self.get_landmarks_in_bbx(image, original_expanded)
        k2 = self.get_landmarks_in_bbx(image, shifted_expanded)
        
        if k1 is None or k2 is None or not self.check_landmark_consistency(k1, k2, bbx):
            return None
            
        # Return relative path from root folder
        return os.path.relpath(img_path, self.root_folder)[:-4]  # Remove .jpg

    def get_landmarks_in_bbx(self, image, bbx):
        """Get landmarks within a specific bounding box region with size checks."""
        x1, y1, x2, y2 = map(int, bbx)
        
        # Ensure the bounding box is valid and large enough
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Check minimum size requirements (SFD detector needs at least 32x32)
        min_size = 32
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
        
        try:
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                return None
                
            # Convert to RGB if needed
            if cropped.shape[2] == 4:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGB)
            elif cropped.shape[2] == 1:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
            else:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
            # Resize if too small (with minimum 32x32)
            if cropped.shape[0] < min_size or cropped.shape[1] < min_size:
                scale_factor = max(min_size/cropped.shape[0], min_size/cropped.shape[1])
                cropped = cv2.resize(cropped, (0,0), fx=scale_factor, fy=scale_factor)
                
            landmarks = self.fa.get_landmarks(cropped)
            if landmarks is None:
                return None
                
            landmarks = landmarks[0]  # Take first face
            # Convert back to original image coordinates
            landmarks[:, 0] = (landmarks[:, 0]/scale_factor if 'scale_factor' in locals() else landmarks[:, 0]) + x1
            landmarks[:, 1] = (landmarks[:, 1]/scale_factor if 'scale_factor' in locals() else landmarks[:, 1]) + y1
            return landmarks
            
        except Exception as e:
            print(f"Landmark detection failed: {str(e)}")
            return None

    def load_state(self):
        """Load processing state from previous run"""
        if os.path.exists(self.state_file):
            state = np.load(self.state_file, allow_pickle=True)
            self.processed_identities = set(state['processed_identities'])
            self.train_data = state['train_data']
            self.val_data = state['val_data']
            self.all_identities = state['all_identities'].tolist()
            self.train_identities = set(state['train_identities'])
            self.val_identities = set(state['val_identities'])
            print(f"Resumed processing state with {len(self.processed_identities)} processed identities")
        else:
            # Initialize identity lists and split
            self.all_identities = [d for d in os.listdir(self.root_folder) 
                                  if os.path.isdir(os.path.join(self.root_folder, d))]
            
            # Create fixed train/val split
            random.seed(self.random_seed)
            random.shuffle(self.all_identities)
            split_idx = int(len(self.all_identities) * (1 - self.val_size))
            self.train_identities = set(self.all_identities[:split_idx])
            self.val_identities = set(self.all_identities[split_idx:])
            
            print(f"Initialized new processing state with {len(self.all_identities)} identities")
            print(f"  - Training identities: {len(self.train_identities)}")
            print(f"  - Validation identities: {len(self.val_identities)}")

    def save_state(self):
        """Save current processing state"""
        np.savez(
            self.state_file,
            processed_identities=list(self.processed_identities),
            train_data=self.train_data,
            val_data=self.val_data,
            all_identities=self.all_identities,
            train_identities=list(self.train_identities),
            val_identities=list(self.val_identities)
        )

    # def process_identity(self, identity):
    #     """Process a single identity and return valid images"""
    #     identity_path = os.path.join(self.root_folder, identity)
    #     valid_images = []
        
    #     for img_name in os.listdir(identity_path):
    #         if not img_name.endswith('.jpg'):
    #             continue
                
    #         img_path = os.path.join(identity_path, img_name)
    #         result = self.process_image(img_path)
            
    #         if result is not None:
    #             valid_images.append(result)
    #             if len(valid_images) >= 5:
    #                 break
        
    #     return valid_images[:5] if len(valid_images) >= 5 else None

    def process_identity(self, identity):
        """Process a single identity and return all valid images"""
        identity_path = os.path.join(self.root_folder, identity)
        valid_images = []
        
        for img_name in os.listdir(identity_path):
            if not img_name.endswith('.jpg'):
                continue
                
            img_path = os.path.join(identity_path, img_name)
            result = self.process_image(img_path)
            
            if result is not None:
                valid_images.append(result)
        
        return valid_images

    def clean_and_split_dataset(self):
        # """Process dataset with incremental train/val updates"""
        # # Track new additions
        # new_train = 0
        # new_val = 0
        
        # # Process identities
        # for identity in tqdm(self.all_identities, desc="Processing identities"):
        #     if identity in self.processed_identities:
        #         continue
                
        #     valid_images = self.process_identity(identity)
        #     if valid_images is None:
        #         # Mark as processed even if invalid to skip in future
        #         self.processed_identities.add(identity)
        #         self.save_state()
        #         continue
                
        #     # Create new row
        #     new_row = np.array(valid_images).reshape(1, 5)
            
        #     try:
        #         # Add to appropriate dataset
        #         if identity in self.train_identities:
        #             if self.train_data.size == 0:
        #                 self.train_data = new_row
        #             else:
        #                 self.train_data = np.vstack((self.train_data, new_row))
        #             new_train += 1
        #         elif identity in self.val_identities:
        #             if self.val_data.size == 0:
        #                 self.val_data = new_row
        #             else:
        #                 self.val_data = np.vstack((self.val_data, new_row))
        #             new_val += 1
        #     except Exception as e:
        #         print(f"Error adding identity {identity}: {str(e)}")
        #         print(f"Train data shape: {self.train_data.shape if hasattr(self.train_data, 'shape') else 'N/A'}")
        #         print(f"Val data shape: {self.val_data.shape if hasattr(self.val_data, 'shape') else 'N/A'}")
        #         print(f"New row shape: {new_row.shape}")
        #         continue
                    
        #     # Update state
        #     self.processed_identities.add(identity)
            
        #     # Save updates after each identity
        #     try:
        #         # Ensure directories exist
        #         os.makedirs(os.path.dirname(self.train_file), exist_ok=True)
        #         os.makedirs(os.path.dirname(self.val_file), exist_ok=True)
                
        #         # Save even if arrays are empty
        #         np.save(self.train_file, self.train_data)
        #         np.save(self.val_file, self.val_data)
        #         self.save_state()
                
        #         print(f"Processed {identity} - {'Train' if identity in self.train_identities else 'Val'}")
        #         print(f"  Train shape: {self.train_data.shape}, Val shape: {self.val_data.shape}")
        #     except Exception as e:
        #         print(f"Error saving state for {identity}: {str(e)}")

        """Process dataset with incremental train/val updates"""
        # Track new additions
        new_train = 0
        new_val = 0
        
        # Process identities
        for identity in tqdm(self.all_identities, desc="Processing identities"):
            if identity in self.processed_identities:
                continue
                
            valid_images = self.process_identity(identity)
            if len(valid_images) < 5:
                # Mark as processed even if invalid to skip in future
                self.processed_identities.add(identity)
                self.save_state()
                continue
                
            # Shuffle valid images to create random groups
            random.shuffle(valid_images)
            
            # Create groups of 5
            num_groups = len(valid_images) // 5
            groups = [
                valid_images[i*5 : (i+1)*5] 
                for i in range(num_groups)
            ]
            
            # Add groups to appropriate dataset
            for group in groups:
                new_row = np.array(group).reshape(1, 5)
                
                try:
                    if identity in self.train_identities:
                        if self.train_data.size == 0:
                            self.train_data = new_row
                        else:
                            self.train_data = np.vstack((self.train_data, new_row))
                        new_train += 1
                    elif identity in self.val_identities:
                        if self.val_data.size == 0:
                            self.val_data = new_row
                        else:
                            self.val_data = np.vstack((self.val_data, new_row))
                        new_val += 1
                except Exception as e:
                    print(f"Error adding identity {identity}: {str(e)}")
                    continue
                    
            # Update state
            self.processed_identities.add(identity)
            
            # Save updates after each identity
            try:
                # Ensure directories exist
                os.makedirs(os.path.dirname(self.train_file), exist_ok=True)
                os.makedirs(os.path.dirname(self.val_file), exist_ok=True)
                
                np.save(self.train_file, self.train_data)
                np.save(self.val_file, self.val_data)
                self.save_state()
                
                print(f"Processed {identity} - {'Train' if identity in self.train_identities else 'Val'}")
                print(f"  Added {len(groups)} groups, Total train: {self.train_data.shape[0]}, Total val: {self.val_data.shape[0]}")
            except Exception as e:
                print(f"Error saving state for {identity}: {str(e)}")
        
        # Final report
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Total processed identities: {len(self.processed_identities)}")
        print(f"New training identities added: {new_train}")
        print(f"New validation identities added: {new_val}")
        print(f"Total training identities: {len(self.train_data)}")
        print(f"Total validation identities: {len(self.val_data)}")
        print(f"Training file: {self.train_file}")
        print(f"Validation file: {self.val_file}")
        
        # Create empty validation file if needed
        if len(self.val_data) == 0:
            print("Warning: Validation dataset is empty!")
            np.save(self.val_file, np.empty((0, 5), dtype=object))

if __name__ == "__main__":
    # Configuration
    ROOT_FOLDER = "ps/scratch/face2d3d/train"
    OUTPUT_DIR = "ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning"
    VAL_SIZE = 0.2  # 20% for validation
    RANDOM_SEED = 42  # Fixed seed for reproducible splits
    
    # Create cleaner and process dataset
    cleaner = VGGFace2DataCleaner(
        root_folder=ROOT_FOLDER,
        output_dir=OUTPUT_DIR,
        use_landmarks_for_bbx=True,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED
    )
    cleaner.clean_and_split_dataset()