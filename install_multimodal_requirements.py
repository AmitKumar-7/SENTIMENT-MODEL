#!/usr/bin/env python3
"""
Installation script for multimodal meme sentiment analysis dependencies
"""

import subprocess
import sys
import pkg_resources
from packaging import version

def install_package(package_name, min_version=None):
    """Install a package using pip"""
    try:
        # Check if package is already installed
        try:
            installed_version = pkg_resources.get_distribution(package_name).version
            if min_version and version.parse(installed_version) < version.parse(min_version):
                print(f"Upgrading {package_name} from {installed_version} to >={min_version}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", f"{package_name}>={min_version}"])
            else:
                print(f"✅ {package_name} {installed_version} already installed")
                return True
        except pkg_resources.DistributionNotFound:
            print(f"Installing {package_name}...")
            package_spec = f"{package_name}>={min_version}" if min_version else package_name
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
            print(f"✅ {package_name} installed successfully")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def install_multimodal_requirements():
    """Install all requirements for multimodal analysis"""
    print("🚀 Installing Multimodal Meme Sentiment Analysis Requirements")
    print("=" * 70)
    
    # Core dependencies
    requirements = [
        # Basic packages
        ("torch", "1.9.0"),
        ("torchvision", "0.10.0"),
        ("transformers", "4.20.0"),
        ("Pillow", "8.0.0"),
        ("opencv-python", "4.5.0"),
        ("numpy", "1.21.0"),
        ("pandas", "1.3.0"),
        ("requests", "2.25.0"),
        ("validators", "0.18.0"),
        
        # OCR
        ("pytesseract", "0.3.8"),
        
        # Multimodal models
        ("timm", "0.6.0"),  # For vision models
        
        # Video processing
        ("moviepy", "1.0.3"),
        ("imageio", "2.19.0"),
        ("imageio-ffmpeg", "0.4.7"),
        
        # Audio processing (optional)
        ("librosa", "0.9.0"),
        ("soundfile", "0.10.0"),
        
        # Additional ML packages
        ("scikit-learn", "1.0.0"),
        ("matplotlib", "3.5.0"),
        ("seaborn", "0.11.0"),
        
        # Utilities
        ("tqdm", "4.62.0"),
        ("packaging", "21.0"),
    ]
    
    # Install packages
    failed_packages = []
    for package_name, min_version in requirements:
        success = install_package(package_name, min_version)
        if not success:
            failed_packages.append(package_name)
    
    print("\n" + "=" * 70)
    print("📋 Installation Summary")
    print("=" * 70)
    
    if failed_packages:
        print(f"❌ Failed to install: {', '.join(failed_packages)}")
        print("\nTry installing manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print("✅ All packages installed successfully!")
    
    # Additional setup instructions
    print("\n🔧 Additional Setup Required:")
    print("-" * 40)
    print("1. Install Tesseract OCR:")
    print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("   Linux: sudo apt-get install tesseract-ocr")
    print("   macOS: brew install tesseract")
    print("\n2. Update your .env file with Tesseract path:")
    print("   TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    print("\n3. For GPU acceleration (optional):")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    return len(failed_packages) == 0

if __name__ == "__main__":
    success = install_multimodal_requirements()
    if success:
        print("\n🎉 Setup complete! You can now run the multimodal analyzer.")
    else:
        print("\n⚠️ Some packages failed to install. Please install them manually.")
        sys.exit(1)
