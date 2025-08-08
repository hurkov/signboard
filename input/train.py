"""
Unified training system for sign language gesture recognition.
Combines image collection, dataset creation, and classifier training.
"""

import os
import time
import asyncio
import threading
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

# Suppress all graphics output before importing any graphics libraries
os.environ['MPLBACKEND'] = 'Agg'  # Use non-interactive matplotlib backend
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'  # Completely suppress OpenCV output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Disable GPU acceleration to prevent graphics
os.environ['DISPLAY'] = ''  # Disable display for headless operation
os.environ['GLOG_minloglevel'] = '3'  # Maximum suppression for Google logging
os.environ['GLOG_logtostderr'] = '0'  # Don't log to stderr
os.environ['GLOG_alsologtostderr'] = '0'  # Don't also log to stderr
os.environ['GLOG_stderrthreshold'] = '3'  # Suppress stderr output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices
# Additional Metal/GPU suppression for macOS
os.environ['MEDIAPIPE_INFERENCE_GPU_BACKEND'] = 'cpu'  # Force CPU backend
os.environ['MEDIAPIPE_DISABLE_GPU_INFERENCE'] = '1'  # Disable GPU inference
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'  # Disable hardware transforms
# Completely suppress Metal/GPU renderer messages on macOS
os.environ['MTL_DEBUG_LAYER'] = '0'
os.environ['MTL_SHADER_VALIDATION'] = '0'
os.environ['MTL_HUD_ENABLED'] = '0'
# Suppress OpenGL context messages
os.environ['GL_SILENCE_ERRORS'] = '1'
# Force MediaPipe to use CPU completely
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['MEDIAPIPE_PREFER_CPU'] = '1'

# Redirect stderr to suppress MediaPipe warnings
import sys
from contextlib import redirect_stderr, redirect_stdout
import io
import subprocess

# Create a null device to suppress output
devnull = io.StringIO()

# Also suppress stdout for complete silence during training
class SilentContext:
    """Context manager to completely suppress all output including C library output"""
    def __init__(self):
        self.original_stderr = sys.stderr
        self.original_stdout = sys.stdout
        # Store original file descriptors
        self.original_stderr_fd = os.dup(2)
        self.original_stdout_fd = os.dup(1)
        
    def __enter__(self):
        # Redirect Python output
        sys.stderr = devnull
        sys.stdout = devnull
        
        # Redirect C library output at the file descriptor level
        try:
            # Open devnull and redirect stderr/stdout file descriptors
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 1)  # stdout
            os.dup2(devnull_fd, 2)  # stderr
            os.close(devnull_fd)
        except:
            pass  # If fd redirection fails, at least Python output is suppressed
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore Python output
        sys.stderr = self.original_stderr
        sys.stdout = self.original_stdout
        
        # Restore C library output
        try:
            os.dup2(self.original_stderr_fd, 2)  # stderr
            os.dup2(self.original_stdout_fd, 1)  # stdout
            os.close(self.original_stderr_fd)
            os.close(self.original_stdout_fd)
        except:
            pass

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False

try:
    # Maximum suppression for MediaPipe import including file descriptor redirection
    old_stderr_fd = os.dup(2)
    old_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    try:
        # Redirect both stdout and stderr at file descriptor level
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        
        # Import MediaPipe with complete silence
        import mediapipe as mp
        MEDIAPIPE_AVAILABLE = True
        
    finally:
        # Restore file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        
except ImportError:
    MEDIAPIPE_AVAILABLE = False
except Exception:
    # If any other error occurs during import, mark as unavailable
    MEDIAPIPE_AVAILABLE = False

# Suppress logging at the Python level too
import logging
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('absl').setLevel(logging.CRITICAL)

# Check if all required dependencies are available
TRAINING_DEPENDENCIES_AVAILABLE = all([
    CV2_AVAILABLE, NUMPY_AVAILABLE, PICKLE_AVAILABLE, MEDIAPIPE_AVAILABLE
])

class SignLanguageTrainer:
    """Unified trainer for sign language gestures"""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent / "data"
        else:
            base_path = Path(base_path)
        
        self.base_path = base_path
        self.images_dir = base_path / "images"
        self.models_dir = base_path / "models"
        self.datasets_dir = base_path / "datasets"
        
        # Create directories
        for dir_path in [self.images_dir, self.models_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Check dependencies
        self.dependencies_available = TRAINING_DEPENDENCIES_AVAILABLE
        
        if self.dependencies_available:
            # MediaPipe setup
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Training parameters
        self.num_samples = 250
        self.collection_delay = 0.1
        
        # State tracking
        self.is_training = False
        self.is_collecting = False
        self.current_samples = 0
        self.collected_data = []
        self.collected_labels = []
        
        # Callbacks for UI updates
        self.progress_callback = None
        self.status_callback = None
        self.completion_callback = None
        
    def set_callbacks(self, progress_callback: Callable = None, 
                     status_callback: Callable = None,
                     completion_callback: Callable = None):
        """Set callback functions for UI updates"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.completion_callback = completion_callback
    
    def _update_status(self, message: str):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(message)
    
    def _update_progress(self, current: int, total: int):
        """Update progress via callback"""
        if self.progress_callback:
            self.progress_callback(current, total)
    
    def _signal_completion(self, success: bool, message: str, model_path: str = None):
        """Signal completion via callback"""
        if self.completion_callback:
            self.completion_callback(success, message, model_path)
    
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check if all required dependencies are available"""
        missing = []
        
        if not CV2_AVAILABLE:
            missing.append("opencv-python")
        if not NUMPY_AVAILABLE:
            missing.append("numpy")
        if not PICKLE_AVAILABLE:
            missing.append("pickle (should be built-in)")
        if not MEDIAPIPE_AVAILABLE:
            missing.append("mediapipe")
        
        return len(missing) == 0, missing
    
    def collect_images_for_gesture(self, gesture_key: str, num_samples: int = None) -> bool:
        """Collect images for a specific gesture using webcam"""
        if not self.dependencies_available:
            self._update_status("❌ Missing dependencies for image collection")
            return False
        
        if num_samples is None:
            num_samples = self.num_samples
        
        self.is_collecting = True
        self.current_samples = 0
        
        # Create directory for this gesture
        gesture_dir = self.images_dir / gesture_key
        gesture_dir.mkdir(exist_ok=True)
        
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self._update_status("❌ Failed to open camera")
                return False
            
            self._update_status(f"📸 Starting image collection for key '{gesture_key}'")
            self._update_status("👋 Show your gesture to the camera...")
            
            # Countdown before starting
            for i in range(3, 0, -1):
                if not self.is_collecting:
                    break
                self._update_status(f"📸 Starting in {i} seconds...")
                time.sleep(1)
            
            if not self.is_collecting:
                cap.release()
                return False
            
            self._update_status("📸 Collection started! Keep showing your gesture...")
            
            sample_count = 0
            while sample_count < num_samples and self.is_collecting:
                ret, frame = cap.read()
                if not ret:
                    self._update_status("❌ Failed to capture frame")
                    break
                
                # Mirror the frame for better user experience
                frame = cv2.flip(frame, 1)
                
                # Save the frame
                img_path = gesture_dir / f"{sample_count:04d}.jpg"
                cv2.imwrite(str(img_path), frame)
                
                sample_count += 1
                self.current_samples = sample_count
                
                # Update progress
                self._update_progress(sample_count, num_samples)
                self._update_status(f"📸 Collecting samples: {sample_count}/{num_samples}")
                
                # Small delay between captures
                time.sleep(self.collection_delay)
            
            cap.release()
            # Note: No cv2.destroyAllWindows() needed since no windows are displayed during training
            
            if sample_count >= num_samples:
                self._update_status(f"✅ Collection complete! {sample_count} samples collected")
                return True
            else:
                self._update_status(f"⚠️ Collection interrupted. {sample_count} samples collected")
                return False
                
        except Exception as e:
            self._update_status(f"❌ Collection failed: {str(e)}")
            return False
        finally:
            self.is_collecting = False
    
    def extract_hand_landmarks(self, image_path: str) -> Optional[List[float]]:
        """Extract hand landmarks from an image using MediaPipe"""
        if not self.dependencies_available:
            return None
        
        try:
            # Maximum environment suppression for this specific operation
            old_display = os.environ.get('DISPLAY', '')
            old_log_level = os.environ.get('OPENCV_LOG_LEVEL', '')
            old_glog_level = os.environ.get('GLOG_minloglevel', '')
            
            # Force maximum suppression
            os.environ['DISPLAY'] = ''
            os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
            os.environ['GLOG_minloglevel'] = '3'
            os.environ['GLOG_stderrthreshold'] = '3'
            
            # Suppress ALL output during MediaPipe processing using enhanced silent context
            with SilentContext():
                # Read image
                img = cv2.imread(image_path)
                if img is None:
                    return None
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe in silent mode with forced CPU backend
                with self.mp_hands.Hands(
                    static_image_mode=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                    max_num_hands=1  # Limit to one hand for faster processing
                ) as hands:
                    
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        # Get first detected hand
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        # Extract landmark coordinates
                        data_aux = []
                        x_coords = []
                        y_coords = []
                        
                        for landmark in hand_landmarks.landmark:
                            x_coords.append(landmark.x)
                            y_coords.append(landmark.y)
                        
                        # Normalize coordinates relative to hand bounding box
                        min_x, min_y = min(x_coords), min(y_coords)
                        for landmark in hand_landmarks.landmark:
                            data_aux.append(landmark.x - min_x)
                            data_aux.append(landmark.y - min_y)
                        
                        return data_aux
                    
                    return None
                
        except Exception as e:
            return None
        finally:
            # Restore environment variables
            if old_display:
                os.environ['DISPLAY'] = old_display
            elif 'DISPLAY' in os.environ:
                del os.environ['DISPLAY']
            
            if old_log_level:
                os.environ['OPENCV_LOG_LEVEL'] = old_log_level
            if old_glog_level:
                os.environ['GLOG_minloglevel'] = old_glog_level
    
    def create_dataset_from_images(self, gesture_key: str) -> bool:
        """Create dataset from collected images"""
        if not self.dependencies_available:
            self._update_status("❌ Missing dependencies for dataset creation")
            return False
        
        try:
            self._update_status("🔄 Processing images and extracting features...")
            
            gesture_dir = self.images_dir / gesture_key
            if not gesture_dir.exists():
                self._update_status(f"❌ No images found for gesture '{gesture_key}'")
                return False
            
            data = []
            labels = []
            
            image_files = list(gesture_dir.glob("*.jpg"))
            total_images = len(image_files)
            
            if total_images == 0:
                self._update_status(f"❌ No image files found in {gesture_dir}")
                return False
            
            processed_count = 0
            valid_count = 0
            
            # Maximum suppression during batch processing
            old_display = os.environ.get('DISPLAY', '')
            old_log_level = os.environ.get('OPENCV_LOG_LEVEL', '')
            old_glog_level = os.environ.get('GLOG_minloglevel', '')
            
            os.environ['DISPLAY'] = ''
            os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
            os.environ['GLOG_minloglevel'] = '3'
            
            try:
                for img_file in image_files:
                    if not self.is_training:
                        break
                    
                    # Process each image with maximum suppression
                    with SilentContext():
                        landmarks = self.extract_hand_landmarks(str(img_file))
                    
                    if landmarks is not None:
                        data.append(landmarks)
                        labels.append(gesture_key)
                        valid_count += 1
                    
                    processed_count += 1
                    
                    # Show progress only
                    progress_percent = (processed_count / total_images) * 100
                    self._update_progress(processed_count, total_images)
                    self._update_status(f"🔄 Analyzing samples... ({progress_percent:.0f}%) - {valid_count} valid")
                    
            finally:
                # Restore environment variables
                if old_display:
                    os.environ['DISPLAY'] = old_display
                elif 'DISPLAY' in os.environ:
                    del os.environ['DISPLAY']
                
                if old_log_level:
                    os.environ['OPENCV_LOG_LEVEL'] = old_log_level
                if old_glog_level:
                    os.environ['GLOG_minloglevel'] = old_glog_level
            
            if not data:
                self._update_status("❌ No valid hand landmarks detected in images")
                return False
            
            # Save dataset
            dataset_path = self.datasets_dir / f"{gesture_key}_dataset.pickle"
            dataset = {
                'data': data,
                'labels': labels,
                'gesture_key': gesture_key,
                'num_samples': len(data)
            }
            
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)
            
            # Store in instance for training
            self.collected_data = data
            self.collected_labels = labels
            
            self._update_status(f"✅ Dataset created: {len(data)} valid samples")
            return True
            
        except Exception as e:
            self._update_status(f"❌ Dataset creation failed: {str(e)}")
            return False
    
    def train_classifier(self, gesture_key: str) -> Optional[str]:
        """Train classifier for the gesture"""
        if not self.dependencies_available:
            self._update_status("❌ Missing dependencies for training")
            return None
        
        try:
            self._update_status("🤖 Training classifier...")
            
            if not self.collected_data or not self.collected_labels:
                self._update_status("❌ No training data available")
                return None
            
            # Convert to numpy arrays
            X = np.asarray(self.collected_data)
            y = np.asarray(self.collected_labels)
            
            # Create a simple model that remembers the gesture pattern
            model_data = {
                'type': 'single_gesture',
                'gesture_key': gesture_key,
                'reference_pattern': np.mean(X, axis=0),
                'threshold': np.std(X, axis=0).mean()
            }
            
            # Save model
            model_path = self.models_dir / f"{gesture_key}_model.pickle"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self._update_status(f"✅ Model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            self._update_status(f"❌ Training failed: {str(e)}")
            return None
    
    def cleanup_training_data(self, gesture_key: str, keep_images: bool = False):
        """Clean up temporary training data"""
        try:
            if not keep_images:
                gesture_dir = self.images_dir / gesture_key
                if gesture_dir.exists():
                    import shutil
                    shutil.rmtree(gesture_dir)
                    self._update_status(f"🧹 Cleaned up images for '{gesture_key}'")
        except Exception as e:
            self._update_status(f"⚠️ Cleanup warning: {str(e)}")
    
    async def train_gesture_async(self, gesture_key: str, num_samples: int = None, 
                                cleanup_images: bool = True) -> bool:
        """Complete async training pipeline for a gesture"""
        if not self.dependencies_available:
            deps_ok, missing = self.check_dependencies()
            if not deps_ok:
                error_msg = f"❌ Missing dependencies: {', '.join(missing)}\n"
                error_msg += "Install with: pip install opencv-python numpy mediapipe"
                self._signal_completion(False, error_msg)
                return False
        
        if self.is_training:
            self._update_status("⚠️ Training already in progress")
            return False
        
        self.is_training = True
        
        try:
            # Step 1: Collect images
            self._update_status(f"🚀 Starting training for key '{gesture_key}'")
            
            def collect_images():
                return self.collect_images_for_gesture(gesture_key, num_samples)
            
            # Use thread executor for camera operations
            with ThreadPoolExecutor() as executor:
                future = executor.submit(collect_images)
                
                # Wait for completion while allowing cancellation
                while not future.done():
                    if not self.is_training:
                        self.stop_training()
                        return False
                    await asyncio.sleep(0.1)
                
                collection_success = future.result()
            
            if not collection_success or not self.is_training:
                self._signal_completion(False, "Image collection failed or was cancelled")
                return False
            
            # Step 2: Create dataset
            if not self.create_dataset_from_images(gesture_key):
                self._signal_completion(False, "Dataset creation failed")
                return False
            
            if not self.is_training:
                self._signal_completion(False, "Training was cancelled")
                return False
            
            # Step 3: Train classifier
            model_path = self.train_classifier(gesture_key)
            if not model_path:
                self._signal_completion(False, "Classifier training failed")
                return False
            
            # Step 4: Cleanup (optional)
            if cleanup_images:
                self.cleanup_training_data(gesture_key, keep_images=False)
            
            # Success
            success_msg = f"✅ Training completed successfully!\n🔑 Gesture for key '{gesture_key}' is ready\n📁 Model saved to: {model_path}"
            self._signal_completion(True, success_msg, model_path)
            return True
            
        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            self._signal_completion(False, error_msg)
            return False
        finally:
            self.is_training = False
    
    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        self.is_collecting = False
        self._update_status("⏹️ Training stopped by user")
    
    def list_trained_gestures(self) -> List[str]:
        """List all trained gestures"""
        try:
            model_files = list(self.models_dir.glob("*_model.pickle"))
            gestures = [f.stem.replace("_model", "") for f in model_files]
            return gestures
        except Exception:
            return []
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics about training data and models"""
        stats = {
            'total_gestures': len(self.list_trained_gestures()),
            'gestures': self.list_trained_gestures(),
            'models_dir': str(self.models_dir),
            'datasets_dir': str(self.datasets_dir),
            'images_dir': str(self.images_dir)
        }
        
        # Add individual gesture stats
        for gesture in stats['gestures']:
            dataset_path = self.datasets_dir / f"{gesture}_dataset.pickle"
            if dataset_path.exists():
                try:
                    with open(dataset_path, 'rb') as f:
                        dataset = pickle.load(f)
                        stats[f'{gesture}_samples'] = dataset.get('num_samples', 0)
                except Exception:
                    stats[f'{gesture}_samples'] = 0
        
        return stats

# Utility functions for integration with main application
def create_trainer(base_path: str = None) -> SignLanguageTrainer:
    """Factory function to create a trainer instance"""
    return SignLanguageTrainer(base_path)

async def train_gesture_for_key(gesture_key: str, trainer: SignLanguageTrainer = None,
                               status_callback: Callable = None,
                               progress_callback: Callable = None,
                               completion_callback: Callable = None) -> bool:
    """Convenience function to train a gesture for a specific key"""
    if trainer is None:
        trainer = create_trainer()
    
    # Set callbacks
    trainer.set_callbacks(progress_callback, status_callback, completion_callback)
    
    # Start training
    return await trainer.train_gesture_async(gesture_key)

if __name__ == "__main__":
    # Test the trainer
    trainer = SignLanguageTrainer()
    
    def status_update(msg):
        print(f"Status: {msg}")
    
    def progress_update(current, total):
        print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
    
    def completion_update(success, message, model_path=None):
        print(f"Completion: {'SUCCESS' if success else 'FAILED'}")
        print(f"Message: {message}")
        if model_path:
            print(f"Model: {model_path}")
    
    trainer.set_callbacks(progress_update, status_update, completion_update)
    
    # Check dependencies
    deps_ok, missing = trainer.check_dependencies()
    print(f"Dependencies available: {deps_ok}")
    if not deps_ok:
        print(f"Missing: {missing}")
    
    # Example usage
    print("Sign Language Trainer Test")
    print("Stats:", trainer.get_training_stats())
