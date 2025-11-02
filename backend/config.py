# Configuration for CycleGAN model

# Model Architecture Parameters
INPUT_SHAPE = (3, 512, 512)  # (channels, height, width)
NUM_RESIDUAL_BLOCKS = 9  # Number of residual blocks in the generator

# Image Processing Parameters
IMAGE_SIZE = 512  # Images will be resized to this size (square)
NORMALIZATION_MEAN = [0.5, 0.5, 0.5]
NORMALIZATION_STD = [0.5, 0.5, 0.5]

# Model Paths (relative to backend directory)
MODEL_DIR = 'models'
G_AB_PATH = f'{MODEL_DIR}/G_AB.pth'  # Real to Sketch (A to B)
G_BA_PATH = f'{MODEL_DIR}/G_BA.pth'  # Sketch to Real (B to A)
D_A_PATH = f'{MODEL_DIR}/D_A.pth'    # Discriminator A (for real images)
D_B_PATH = f'{MODEL_DIR}/D_B.pth'    # Discriminator B (for sketches)

# Detection Thresholds
COLOR_VARIANCE_THRESHOLD = 1000  # Lower values indicate sketches
EDGE_DENSITY_THRESHOLD = 0.15    # Higher values indicate sketches
BRIGHTNESS_THRESHOLD = 200       # Used with edge density

# Discriminator-based Detection Thresholds
DISCRIMINATOR_THRESHOLD = 0.1    # Minimum score difference to make a decision
DISCRIMINATOR_CONFIDENCE = 0.6   # Minimum confidence score for strong classification

# Output Configuration
OUTPUT_DIR = 'generated_images'  # Directory to store generated images
SAVE_INPUT_IMAGES = True         # Whether to save input images as well
SAVE_IMAGES = True               # Master switch for saving images

# Server Configuration
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True
