from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
import cv2
import os
import config

app = Flask(__name__)
CORS(app)

# Define the Generator architecture (matching your trained model)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features), 
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(GeneratorResNet, self).__init__()
        
        channels = input_shape[0]
        
        # Initial Convolution Block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
        
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]
            
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2), # --> width*2, heigh*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            
        # Output Layer
        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()
                 ]
        
        # Unpacking
        self.model = nn.Sequential(*model) 
        
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        channels, height, width = input_shape
        
        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height//2**4, width//2**4)
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
        
    def forward(self, img):
        return self.model(img)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory for generated images
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, config.OUTPUT_DIR)
if config.SAVE_IMAGES and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Load configuration
input_shape = config.INPUT_SHAPE
num_residual_blocks = config.NUM_RESIDUAL_BLOCKS

G_AB = GeneratorResNet(input_shape=input_shape, num_residual_block=num_residual_blocks).to(device)
G_BA = GeneratorResNet(input_shape=input_shape, num_residual_block=num_residual_blocks).to(device)

# Initialize discriminators for image type detection
D_A = Discriminator(input_shape=input_shape).to(device)  # Discriminator for domain A (real images)
D_B = Discriminator(input_shape=input_shape).to(device)  # Discriminator for domain B (sketches)

# Load the trained weights
try:
    g_ab_path = os.path.join(script_dir, config.G_AB_PATH)
    g_ba_path = os.path.join(script_dir, config.G_BA_PATH)
    d_a_path = os.path.join(script_dir, config.D_A_PATH)
    d_b_path = os.path.join(script_dir, config.D_B_PATH)
    
    print(f"Loading models from:")
    print(f"  G_AB: {g_ab_path}")
    print(f"  G_BA: {g_ba_path}")
    print(f"  D_A: {d_a_path}")
    print(f"  D_B: {d_b_path}")
    
    G_AB.load_state_dict(torch.load(g_ab_path, map_location=device))
    G_BA.load_state_dict(torch.load(g_ba_path, map_location=device))
    D_A.load_state_dict(torch.load(d_a_path, map_location=device))
    D_B.load_state_dict(torch.load(d_b_path, map_location=device))
    
    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please check that the model files exist and match the architecture.")
    print(f"Expected paths:")
    print(f"  {g_ab_path}")
    print(f"  {g_ba_path}")
    print(f"  {d_a_path}")
    print(f"  {d_b_path}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORMALIZATION_MEAN, std=config.NORMALIZATION_STD)
])

def detect_image_type(image):
    """
    Detect if image is a sketch or real face using discriminators.
    The discriminators output higher values for real images in their domain.
    D_A was trained on real images (domain A)
    D_B was trained on sketches (domain B)
    Returns: 'sketch' or 'real'
    """
    # Preprocess the image
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        # Get discriminator scores
        d_a_score = torch.sigmoid(D_A(input_tensor)).mean().item()  # Score for being a real image
        d_b_score = torch.sigmoid(D_B(input_tensor)).mean().item()  # Score for being a sketch
        
        print(f"Discriminator scores - D_A (real): {d_a_score:.4f}, D_B (sketch): {d_b_score:.4f}")
        
        # Also use traditional CV methods as backup
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Check color variance (sketches are typically grayscale or low color variance)
        color_variance = np.var(img_array)
        
        # Check edge density (sketches have high edge density)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate brightness mean
        brightness_mean = np.mean(gray)
        
        print(f"CV features - Color variance: {color_variance:.2f}, Edge density: {edge_density:.4f}, Brightness: {brightness_mean:.2f}")
        
        # Decision logic: prioritize discriminator scores, use CV features as backup
        # If D_A score is significantly higher, it's likely a real image
        # If D_B score is significantly higher, it's likely a sketch
        
        score_diff = d_a_score - d_b_score
        
        # Combined decision with discriminator having higher weight
        is_real = (
            (score_diff > config.DISCRIMINATOR_THRESHOLD) or  # D_A strongly indicates real image
            (d_a_score > config.DISCRIMINATOR_CONFIDENCE and color_variance > config.COLOR_VARIANCE_THRESHOLD)  # High D_A score + colored
        )
        
        is_sketch = (
            (score_diff < -config.DISCRIMINATOR_THRESHOLD) or  # D_B strongly indicates sketch
            (d_b_score > config.DISCRIMINATOR_CONFIDENCE and (color_variance < config.COLOR_VARIANCE_THRESHOLD or 
                                  (edge_density > config.EDGE_DENSITY_THRESHOLD and brightness_mean > config.BRIGHTNESS_THRESHOLD)))
        )
        
        # If both or neither, use discriminator scores as tiebreaker
        if is_real and not is_sketch:
            return 'real'
        elif is_sketch and not is_real:
            return 'sketch'
        else:
            # Use discriminator scores as final decision
            return 'real' if d_a_score > d_b_score else 'sketch'

def preprocess_image(image):
    """Preprocess image for model input"""
    return transform(image).unsqueeze(0).to(device)

def postprocess_image(tensor):
    """Convert tensor back to PIL Image"""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    return image

def save_generated_image(input_image, output_image, conversion, timestamp):
    """Save the generated image and optionally the input image"""
    if not config.SAVE_IMAGES:
        return None
    
    try:
        # Create timestamp-based filename
        from datetime import datetime
        
        # Save output image
        output_filename = f"{timestamp}_{conversion}_output.png"
        output_path = os.path.join(output_dir, output_filename)
        output_image.save(output_path)
        print(f"Saved generated image: {output_filename}")
        
        # Save input image if configured
        if config.SAVE_INPUT_IMAGES:
            input_filename = f"{timestamp}_{conversion}_input.png"
            input_path = os.path.join(output_dir, input_filename)
            input_image.save(input_path)
            print(f"Saved input image: {input_filename}")
        
        return output_filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'device': str(device)})

@app.route('/convert', methods=['POST'])
def convert_image():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Generate timestamp for this conversion
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Detect image type
        image_type = detect_image_type(image)
        print(f"Detected image type: {image_type}")
        
        # Preprocess
        input_tensor = preprocess_image(image)
        
        # Select appropriate generator
        with torch.no_grad():
            if image_type == 'sketch':
                # Sketch to Real (B to A)
                output_tensor = G_BA(input_tensor)
                conversion = 'sketch_to_real'
            else:
                # Real to Sketch (A to B)
                output_tensor = G_AB(input_tensor)
                conversion = 'real_to_sketch'
        
        # Postprocess
        output_image = postprocess_image(output_tensor)
        
        # Save images to disk
        saved_filename = save_generated_image(image, output_image, conversion, timestamp)
        
        # Convert to base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response_data = {
            'output_image': f'data:image/png;base64,{img_str}',
            'detected_type': image_type,
            'conversion': conversion
        }
        
        # Add saved filename to response if image was saved
        if saved_filename:
            response_data['saved_as'] = saved_filename
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
