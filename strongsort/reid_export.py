# reid_export.py

class export_formats:
    def __init__(self):
        # Define supported model file suffixes in the expected order:
        # PyTorch (.pt), TorchScript (.jit), ONNX (.onnx), OpenVINO (.xml),
        # TensorRT (.engine), and TensorFlow Lite (.tflite)
        self.Suffix = ('.pt', '.jit', '.onnx', '.xml', '.engine', '.tflite')

