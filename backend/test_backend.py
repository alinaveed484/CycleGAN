"""
Test script to verify the backend server is working correctly
"""
import requests
import sys
from pathlib import Path

def test_health():
    """Test if the server is running"""
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is healthy")
            print(f"  Device: {data.get('device', 'Unknown')}")
            return True
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure it's running on port 5000")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_convert(image_path):
    """Test image conversion"""
    if not Path(image_path).exists():
        print(f"✗ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://localhost:5000/convert', files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Image converted successfully")
            print(f"  Detected type: {data.get('detected_type', 'Unknown')}")
            print(f"  Conversion: {data.get('conversion', 'Unknown')}")
            
            # Save output image
            output_image = data.get('output_image', '')
            if output_image:
                print(f"  Output image data length: {len(output_image)} characters")
            return True
        else:
            print(f"✗ Conversion failed with status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("CycleGAN Backend Test Script")
    print("=" * 50)
    
    # Test 1: Health check
    print("\nTest 1: Health Check")
    health_ok = test_health()
    
    if not health_ok:
        print("\n⚠ Server is not running. Please start it with: python app.py")
        sys.exit(1)
    
    # Test 2: Image conversion (if image path provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nTest 2: Image Conversion")
        print(f"Testing with image: {image_path}")
        test_convert(image_path)
    else:
        print("\nTo test image conversion, run:")
        print("  python test_backend.py <path_to_image>")
    
    print("\n" + "=" * 50)
