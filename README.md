# CycleGAN Sketch ↔ Real Face Converter

This application uses a trained CycleGAN model to convert between sketches and real faces automatically.

## Features
- Upload images or use live camera input
- Automatic detection of input type (sketch vs real face)
- Bidirectional conversion: sketch → real face OR real face → sketch
- Interactive UI with animated elements

## Project Structure
```
CycleGAN/
├── backend/              # Flask server for PyTorch inference
│   ├── app.py           # Main Flask application
│   ├── requirements.txt # Python dependencies
│   └── README.md
├── vite-project/        # React frontend
│   ├── src/
│   │   └── App.jsx     # Main application component
│   ├── models/         # Trained PyTorch models
│   │   ├── G_AB.pth    # Generator A→B (sketch→real)
│   │   ├── G_BA.pth    # Generator B→A (real→sketch)
│   │   ├── D_A.pth     # Discriminator A
│   │   └── D_B.pth     # Discriminator B
│   └── package.json
└── README.md           # This file
```

## Setup Instructions

### 1. Backend Setup (Python/Flask)

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **IMPORTANT**: Adjust the Generator architecture in `app.py` if needed to match your trained model's architecture. The current implementation uses a standard CycleGAN architecture with 9 residual blocks.

5. Start the Flask server:
```bash
python app.py
```

The backend server will run on `http://localhost:5000`

### 2. Frontend Setup (React/Vite)

1. Navigate to the vite-project directory:
```bash
cd vite-project
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## Usage

1. Make sure both backend (port 5000) and frontend (port 5173) servers are running
2. Open your browser to `http://localhost:5173`
3. Upload an image or use the camera to capture one
4. Click "Generate Image"
5. The system will:
   - Automatically detect if the input is a sketch or real face
   - Convert sketch → real face OR real → sketch accordingly
   - Display the result in the output section

## How It Works

### Image Detection Algorithm
The backend automatically detects whether an input image is a sketch or real face based on:
- **Color variance**: Sketches typically have low color variance
- **Edge density**: Sketches have high edge density
- **Brightness**: Sketches often have bright backgrounds

### Model Selection
- If **sketch** is detected → uses `G_AB` (Generator A→B) to convert to real face
- If **real face** is detected → uses `G_BA` (Generator B→A) to convert to sketch

## Troubleshooting

### Backend Issues

1. **Model loading errors**: 
   - Ensure the Generator architecture in `app.py` matches your trained model
   - Check that model files are in the correct location: `vite-project/models/`

2. **CUDA/GPU errors**:
   - The code automatically falls back to CPU if CUDA is not available
   - For CPU-only usage, make sure you install PyTorch CPU version

3. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

### Frontend Issues

1. **CORS errors**:
   - The backend already includes CORS support via `flask-cors`
   - Make sure the backend is running on port 5000

2. **Connection refused**:
   - Verify the backend server is running
   - Check that the fetch URL in App.jsx is correct: `http://localhost:5000/convert`

## Customization

### Adjusting the Generator Architecture
If you used a different architecture for your CycleGAN, modify the `Generator` class in `backend/app.py`:
- Change the number of residual blocks
- Adjust downsampling/upsampling layers
- Modify channel dimensions

### Tuning Detection Algorithm
Adjust the thresholds in the `detect_image_type()` function in `backend/app.py`:
```python
is_sketch = (
    (color_variance < 1000) or  # Adjust this threshold
    (edge_density > 0.15 and brightness_mean > 200)  # Adjust these
)
```

### Changing Display Time
To adjust how long the "Why did the ducks cross the road?" text appears, modify the timeout in `App.jsx`:
```javascript
setTimeout(() => setShowDuckLine(false), 3000);  // Change 3000 (ms)
```

## Technologies Used
- **Backend**: Flask, PyTorch, OpenCV, Pillow
- **Frontend**: React, Vite, Framer Motion, Tailwind CSS
- **Models**: CycleGAN (PyTorch)

## License
MIT
