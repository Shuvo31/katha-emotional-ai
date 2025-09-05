#!/usr/bin/env python3
"""
Download required models for Katha
Compatible with Python 3.13
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_transformers_models():
    """Download required transformer models"""
    try:
        from transformers import AutoTokenizer, AutoModel, pipeline
        
        models_to_download = [
            "j-hartmann/emotion-english-distilroberta-base",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        ]
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        for model_name in models_to_download:
            logger.info(f"Downloading {model_name}...")
            try:
                # Download tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(models_dir)
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=str(models_dir)
                )
                
                # Test the model works
                classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    device=-1  # CPU only for compatibility
                )
                
                # Test with sample text
                test_result = classifier("I am happy")
                logger.info(f"✅ {model_name} downloaded and tested successfully")
                logger.info(f"   Test result: {test_result}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {model_name}: {e}")
                continue
                
    except ImportError as e:
        logger.error(f"Transformers not available: {e}")
        return False
    
    return True

def download_tts_models():
    """Download TTS models if available"""
    try:
        from TTS.api import TTS
        
        # Try to download a basic English TTS model
        logger.info("Downloading TTS model...")
        
        # Use CPU-compatible model
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)
        logger.info("✅ TTS model downloaded successfully")
        
        # Test the model
        test_audio = tts.tts("Hello, this is a test.")
        if test_audio is not None and len(test_audio) > 0:
            logger.info("✅ TTS model tested successfully")
        
        return True
        
    except ImportError:
        logger.warning("TTS library not available, skipping TTS model download")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to download TTS model: {e}")
        return False

def check_system_requirements():
    """Check system requirements"""
    logger.info("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    
    # Check available packages
    required_packages = [
        "torch", "transformers", "numpy", "pandas", 
        "streamlit", "plotly", "soundfile", "librosa"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    return True

def create_model_info():
    """Create model information file"""
    model_info = {
        "emotion_models": {
            "j-hartmann/emotion-english-distilroberta-base": {
                "description": "Emotion classification model for English text",
                "emotions": ["joy", "sadness", "anger", "fear", "surprise", "disgust"],
                "language": "en"
            },
            "cardiffnlp/twitter-xlm-roberta-base-sentiment": {
                "description": "Multilingual sentiment analysis model",
                "languages": ["en", "bn", "hi", "multiple"],
                "sentiments": ["positive", "negative", "neutral"]
            }
        },
        "tts_models": {
            "tts_models/en/ljspeech/tacotron2-DDC": {
                "description": "English TTS model based on Tacotron2",
                "language": "en",
                "voice": "female"
            }
        },
        "download_date": None,
        "status": "pending"
    }
    
    import json
    from datetime import datetime
    
    model_info["download_date"] = datetime.now().isoformat()
    
    with open("models/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("✅ Model information file created")

def main():
    """Main function to download all required models"""
    logger.info("🤖 Starting Katha model download process...")
    
    # Check system requirements first
    if not check_system_requirements():
        logger.error("❌ System requirements not met. Exiting.")
        sys.exit(1)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_downloads = 2
    
    # Download transformer models
    logger.info("\n📦 Downloading Transformer models...")
    if download_transformers_models():
        success_count += 1
        logger.info("✅ Transformer models download completed")
    else:
        logger.error("❌ Transformer models download failed")
    
    # Download TTS models
    logger.info("\n🔊 Downloading TTS models...")
    if download_tts_models():
        success_count += 1
        logger.info("✅ TTS models download completed")
    else:
        logger.warning("⚠️ TTS models download skipped or failed")
    
    # Create model info file
    create_model_info()
    
    # Summary
    logger.info(f"\n📊 Download Summary:")
    logger.info(f"   ✅ Successful downloads: {success_count}/{total_downloads}")
    
    if success_count == total_downloads:
        logger.info("🎯 All models downloaded successfully!")
        
        # Update model info
        import json
        with open("models/model_info.json", "r") as f:
            model_info = json.load(f)
        model_info["status"] = "completed"
        with open("models/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
            
    elif success_count > 0:
        logger.warning("⚠️ Some models downloaded successfully, others failed")
        logger.warning("The system should still work with reduced functionality")
    else:
        logger.error("❌ No models downloaded successfully")
        logger.error("Please check your internet connection and package installations")
    
    logger.info("\n🎭 Model download process completed!")
    logger.info("You can now run: streamlit run demo/streamlit_app.py")

if __name__ == "__main__":
    main()