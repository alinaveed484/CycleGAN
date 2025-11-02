import { useState, useRef } from "react";
import { Upload, Camera, ArrowRight } from "lucide-react";
import { Toaster, toast } from "react-hot-toast";
import { motion } from "framer-motion";
import BouncingShape from "./BouncingShape";
import AvoidingShape from "./AvoidingShape";
import DuckFamily from "./DuckFamily";

const movingObjects = [
  {
    id: 1,
    style: { width: 100, height: 100, top: '5%', left: '10%', background: '#6aaeb4', borderRadius: '30% 70% 70% 30% / 30% 30% 70% 70%' },
    animate: { y: [0, -40, 0], x: [0, 60, 0] },
    transition: { duration: 8, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' },
  },
  {
    id: 2,
    style: { width: 60, height: 60, top: '70%', left: '80%', background: '#3e757a', borderRadius: '50%' },
    animate: { y: [0, 50, 0], x: [0, -80, 0] },
    transition: { duration: 12, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' },
  },
  {
    id: 3,
    style: { width: 80, height: 80, top: '80%', left: '15%', background: '#aed3d6', borderRadius: '10px' },
    animate: { rotate: [0, -360, 0], x: [0, 100, 0] },
    transition: { duration: 15, repeat: Infinity, ease: 'linear' },
  },
  {
    id: 4,
    style: { width: 40, height: 40, top: '15%', left: '75%', background: '#1b3436', borderRadius: '50%' },
    animate: { y: [0, -70, 0], x: [0, 70, 0] },
    transition: { duration: 9, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' },
  },
  {
    id: 5,
    style: { width: 120, height: 80, top: '40%', left: '40%', background: '#6aaeb4', borderRadius: '80% 20% 80% 20% / 50% 50% 50% 50%' },
    animate: { y: [0, 60, 0], x: [0, -60, 0] },
    transition: { duration: 10, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' },
  },
  {
    id: 6,
    style: { width: 50, height: 50, top: '90%', left: '50%', background: '#3e757a' },
    animate: { y: [0, -100, 0], x: [0, 100, 0] },
    transition: { duration: 14, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' },
  },
  {
    id: 7,
    style: { width: 70, height: 70, top: '5%', left: '90%', background: '#aed3d6', borderRadius: '20px' },
    animate: { rotate: [0, 360, 0], y: [0, 120, 0] },
    transition: { duration: 18, repeat: Infinity, ease: 'linear' },
  },
  {
    id: 8,
    style: { width: 30, height: 30, top: '50%', left: '5%', background: '#1b3436', borderRadius: '50%' },
    animate: { y: [0, 80, 0], x: [0, -80, 0] },
    transition: { duration: 7, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' },
  },
  {
    id: 9,
    style: { width: 90, height: 40, top: '25%', left: '25%', background: '#3e757a', borderRadius: '50px' },
    animate: { y: [0, -90, 0], x: [0, 90, 0] },
    transition: { duration: 11, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' },
  },
  {
    id: 10,
    style: { width: 60, height: 60, top: '75%', left: '60%', background: '#6aaeb4', transform: 'skew(15deg, 15deg)' },
    animate: { rotate: [0, -360, 0], y: [0, -110, 0] },
    transition: { duration: 16, repeat: Infinity, ease: 'linear' },
  },
];

function App() {
  const [showDuckLine, setShowDuckLine] = useState(false);

  // Callback for when ducks leave the screen
  const handleDucksLeave = () => {
    setShowDuckLine(true);
    setTimeout(() => setShowDuckLine(false), 3000);
  };
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileBlob, setSelectedFileBlob] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [detectedType, setDetectedType] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const constraintsRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(URL.createObjectURL(file));
      setSelectedFileBlob(file);
      setOutputImage(null);
      setDetectedType(null);
      toast.success("File selected!");
    }
  };

  const handleCameraClick = async () => {
    setIsCameraOpen(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    } catch (error) {
      console.error("Error accessing camera:", error);
      toast.error("Could not access camera.");
      setIsCameraOpen(false);
    }
  };

  const handleTakePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    
    // Convert canvas to blob
    canvas.toBlob((blob) => {
      const dataUrl = canvas.toDataURL("image/png");
      setSelectedFile(dataUrl);
      setSelectedFileBlob(blob);
      setOutputImage(null);
      setDetectedType(null);
      setIsCameraOpen(false);
      video.srcObject.getTracks().forEach((track) => track.stop());
      toast.success("Photo taken!");
    }, "image/png");
  };

  const handleSubmit = async () => {
    if (!selectedFile || !selectedFileBlob) {
      toast.error("Please select a file or take a photo first.");
      return;
    }

    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('image', selectedFileBlob);

      const response = await fetch('http://localhost:5000/convert', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process image');
      }

      const data = await response.json();
      
      setOutputImage(data.output_image);
      setDetectedType(data.detected_type);
      
      toast.success(
        `Detected as ${data.detected_type}. Converted to ${
          data.conversion === 'sketch_to_real' ? 'real face' : 'sketch'
        }!`
      );
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to process image. Make sure the backend server is running.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#ddedee] text-[#1b3436] relative overflow-hidden" ref={constraintsRef}>
      <div className="absolute inset-0 z-0">
        {movingObjects.map((obj) => (
          <motion.div
            key={obj.id}
            className="absolute"
            style={obj.style}
            animate={obj.animate}
            transition={obj.transition}
          />
        ))}
        <BouncingShape constraintsRef={constraintsRef} />
        <AvoidingShape />
  <DuckFamily onDucksLeave={handleDucksLeave} />
      </div>
  <div className="relative z-10">
        <Toaster position="top-center" reverseOrder={false} />
        <header className="py-4 px-6 border-b border-[#aed3d6] flex justify-center bg-white/30 backdrop-blur-sm">
          <h1 className="text-2xl font-bold">Image-to-Image</h1>
        </header>
        <main className="flex flex-col md:flex-row gap-8 p-8">
          {/* Input Section */}
          <div className="flex-1 flex flex-col gap-4">
            <h2 className="text-xl font-semibold">Input</h2>
            <div className="flex-1 rounded-lg border-2 border-dashed border-[#aed3d6] flex items-center justify-center p-8 bg-white/30 backdrop-blur-sm">
              {selectedFile ? (
                <img
                  src={selectedFile}
                  alt="Selected"
                  className="max-h-full max-w-full rounded-lg"
                />
              ) : isCameraOpen ? (
                <div className="flex flex-col items-center gap-4">
                  <video ref={videoRef} autoPlay className="w-full rounded-lg" />
                  <button
                    onClick={handleTakePhoto}
                    className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-[#ddedee] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1b3436] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-[#3e757a] text-[#ddedee] hover:bg-[#3e757a]/90 h-10 px-4 py-2"
                  >
                    <Camera className="mr-2 h-4 w-4" />
                    Take Photo
                  </button>
                </div>
              ) : (
                <div className="text-center">
                  <p className="mb-4">Upload a file or use your camera</p>
                  <div className="flex gap-4 justify-center">
                    <label className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-[#ddedee] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1b3436] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-[#3e757a] text-[#ddedee] hover:bg-[#3e757a]/90 h-10 px-4 py-2 cursor-pointer">
                      <Upload className="mr-2 h-4 w-4" />
                      Upload File
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </label>
                    <button
                      onClick={handleCameraClick}
                      className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-[#ddedee] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1b3436] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-[#6aaeb4] text-[#1b3436] hover:bg-[#6aaeb4]/80 h-10 px-4 py-2"
                    >
                      <Camera className="mr-2 h-4 w-4" />
                      Use Camera
                    </button>
                  </div>
                </div>
              )}
            </div>
            <button
              onClick={handleSubmit}
              disabled={isProcessing || !selectedFile}
              className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-[#ddedee] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1b3436] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-[#3e757a] text-[#ddedee] hover:bg-[#3e757a]/90 h-10 px-4 py-2 w-full"
            >
              {isProcessing ? (
                <>
                  <span className="mr-2">Processing...</span>
                  <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                </>
              ) : (
                <>
                  Generate Image
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </button>
            {detectedType && (
              <div className="text-sm text-center text-[#1b3436] bg-[#aed3d6]/30 p-2 rounded">
                Detected: <span className="font-semibold capitalize">{detectedType}</span>
              </div>
            )}
          </div>

          {/* Output Section */}
          <div className="flex-1 flex flex-col gap-4">
            <h2 className="text-xl font-semibold">Output</h2>
            <div className="flex-1 rounded-lg border-2 border-dashed border-[#aed3d6] flex items-center justify-center p-8 bg-white/30 backdrop-blur-sm">
              {outputImage ? (
                <img
                  src={outputImage}
                  alt="Output"
                  className="max-h-full max-w-full rounded-lg"
                />
              ) : (
                <div className="text-center">
                  <p>The generated image will appear here</p>
                </div>
              )}
            </div>
          </div>
        </main>
        <canvas ref={canvasRef} className="hidden" />
      </div>
      {/* Conditional bottom center line */}
      {showDuckLine && (
        <div style={{
          position: 'fixed',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(255,255,255,0.8)',
          padding: '4px 12px',
          borderRadius: '8px',
          fontSize: '1rem',
          color: '#333',
          zIndex: 1000
        }}>
          Why did the ducks cross the Screen?
        </div>
      )}
    </div>
  );
}

export default App;