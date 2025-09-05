#!/bin/bash

echo "🎭 Setting up Katha (কথা) development environment for Python 3.13..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed and get version
print_status "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
    
    # Check if Python version is 3.8 or higher
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_success "Python version is compatible (>=3.8)"
    else
        print_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Please run this script from the project root directory."
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment 'katha_env'..."
python3 -m venv katha_env

if [ $? -eq 0 ]; then
    print_success "Virtual environment created successfully"
else
    print_error "Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source katha_env/Scripts/activate
    print_success "Virtual environment activated (Windows)"
else
    # Linux/Mac
    source katha_env/bin/activate
    print_success "Virtual environment activated (Unix/Linux)"
fi

# Upgrade pip, setuptools, and wheel
print_status "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first (for better compatibility)
print_status "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    print_status "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install dependencies
print_status "Installing project dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Dependencies installed successfully"
else
    print_error "Failed to install some dependencies"
    print_warning "This might be due to Python 3.13 compatibility issues with some packages"
    print_status "Attempting to install with --no-deps flag for problematic packages..."
    
    # Try installing problematic packages individually
    pip install transformers --no-deps
    pip install librosa --no-deps
    pip install TTS --no-deps
    
    # Install their dependencies
    pip install tokenizers datasets accelerate
    pip install numba resampy pooch audioread
    pip install coqui-ai-TTS
fi

# Install the project in development mode
print_status "Installing Katha in development mode..."
pip install -e .

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p demo/assets/{audio_samples,images}
mkdir -p data/{cultural_contexts,emotion_mappings,sample_texts}
mkdir -p logs

# Download required models (if script exists)
if [ -f "scripts/download_models.py" ]; then
    print_status "Downloading required ML models..."
    python scripts/download_models.py
else
    print_warning "Model download script not found. Models will be downloaded on first use."
fi

# Create sample data files
print_status "Creating sample data files..."

# Sample cultural context data
cat > data/cultural_contexts/bengali_contexts.json << 'EOF'
{
  "formality_markers": {
    "high": ["আপনি", "দাদা", "দিদি", "স্যার", "ম্যাডাম"],
    "medium": ["তুমি"],
    "low": ["তুই"]
  },
  "emotion_words": {
    "joy": ["খুশি", "আনন্দ", "দারুণ", "চমৎকার"],
    "sadness": ["দুঃখ", "কষ্ট", "মন খারাপ"],
    "anger": ["রাগ", "ক্রোধ", "বিরক্ত"]
  }
}
EOF

cat > data/cultural_contexts/hindi_contexts.json << 'EOF'
{
  "formality_markers": {
    "high": ["आप", "जी", "साहब", "श्रीमान"],
    "medium": ["तुम"],
    "low": ["तू"]
  },
  "emotion_words": {
    "joy": ["खुश", "खुशी", "आनंद", "शानदार"],
    "sadness": ["दुःख", "गम", "उदास"],
    "anger": ["गुस्सा", "क्रोध", "नाराज"]
  }
}
EOF

# Create a simple test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify Katha installation
"""

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
        
        print("\n🎭 All core dependencies imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_katha_modules():
    """Test if Katha modules can be imported"""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # These will fail initially but that's expected
        try:
            from src.language_detector import KathaLanguageDetector
            print("✅ Language Detector module")
        except:
            print("⏳ Language Detector module (not implemented yet)")
            
        try:
            from src.emotion_analyzer import KathaEmotionAnalyzer
            print("✅ Emotion Analyzer module")
        except:
            print("⏳ Emotion Analyzer module (not implemented yet)")
            
        return True
        
    except Exception as e:
        print(f"❌ Module test error: {e}")
        return False

if __name__ == "__main__":
    print("🎭 Testing Katha Installation...\n")
    
    imports_ok = test_imports()
    modules_ok = test_katha_modules()
    
    if imports_ok:
        print("\n🎯 Installation test completed!")
        print("\n📝 Next steps:")
        print("   1. Implement the core modules in src/")
        print("   2. Run: streamlit run demo/streamlit_app.py")
        print("   3. Start building your emotional AI voice!")
    else:
        print("\n❌ Installation has issues. Please check the error messages above.")
EOF

# Run the test
print_status "Running installation test..."
python test_installation.py

# Final instructions
print_success "🎯 Katha development environment setup completed!"
echo ""
echo "📝 Quick Start Commands:"
echo "   # Activate environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source katha_env/Scripts/activate"
else
    echo "   source katha_env/bin/activate"
fi
echo ""
echo "   # Run demo (once implemented):"
echo "   streamlit run demo/streamlit_app.py"
echo ""
echo "   # Run tests:"
echo "   python test_installation.py"
echo ""
echo "🎭 Ready to build emotional AI voices for Indian languages!"

# Clean up test file
rm -f test_installation.py