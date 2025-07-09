# Sign Language Keyboard 🤟⌨️

A Python-based application that converts sign language gestures into keyboard input using computer vision and machine learning.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Requirements](#requirements)
- [License](#license)

## 🎯 Overview

This application uses real-time computer vision to detect and interpret sign language gestures, converting them into keyboard input. Perfect for accessibility applications and hands-free computer interaction.

## ✨ Features

- ✅ Real-time sign language gesture recognition
- ✅ Customizable gesture templates
- ✅ Cross-platform keyboard simulation
- ✅ Comprehensive system compatibility checks
- ✅ Modular architecture for easy extension
- ✅ Error handling and logging system

## 📁 Project Structure

```
sign_language_keyboard/
├── main.py                    # Main entry point and application launcher
├── config/
│   ├── __init__.py
│   ├── settings.py           # Configuration constants and settings
│   └── requirements.txt      # Project dependencies
├── core/
│   ├── __init__.py
│   ├── compatibility_checker.py  # System compatibility validation
│   ├── system_checker.py         # Screen resolution, files, webcam checks
│   └── template_manager.py       # Template operations and management
├── input/
│   ├── __init__.py
│   ├── camera_handler.py         # Webcam operations and video capture
│   └── gesture_detector.py       # Hand tracking and gesture recognition
├── output/
│   ├── __init__.py
│   └── keyboard_simulator.py     # Keyboard input simulation
├── templates/                    # Gesture templates storage
│   ├── gestures/                 # Individual gesture template files
│   └── mappings/                 # Gesture-to-key mapping configurations
└── utils/
    ├── __init__.py
    └── logger.py                 # Error handling and logging utilities
```

### 📂 Directory Descriptions

#### `config/`
Contains application configuration files and dependencies:
- **`settings.py`**: Application constants, default values, and configuration parameters
- **`requirements.txt`**: Python package dependencies

#### `core/`
Core functionality and system validation:
- **`compatibility_checker.py`**: Validates Python version and system compatibility
- **`system_checker.py`**: Checks screen resolution, file integrity, and webcam access
- **`template_manager.py`**: Manages gesture templates and validates template folder

#### `input/`
Input processing modules:
- **`camera_handler.py`**: Handles webcam initialization, frame capture, and video processing
- **`gesture_detector.py`**: Implements hand tracking, landmark detection, and gesture recognition

#### `output/`
Output generation modules:
- **`keyboard_simulator.py`**: Simulates keyboard input based on recognized gestures

#### `templates/`
Template storage system:
- **`gestures/`**: Stores individual gesture template files
- **`mappings/`**: Contains gesture-to-keyboard mapping configurations

#### `utils/`
Utility functions:
- **`logger.py`**: Centralized logging and error handling system

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera access
- Operating System: Windows 10+, macOS 10.14+, or Linux

### Setup Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd sign_language_keyboard
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r config/requirements.txt
```

4. **Verify installation:**
```bash
python main.py
```

## 💻 Usage

### Basic Usage
```bash
python main.py
```

### First Run
The application will perform the following initialization checks:
1. ✅ System compatibility verification
2. ✅ Screen resolution detection
3. ✅ Required files validation
4. ✅ Template folder verification
5. ✅ Webcam access confirmation

### Adding Custom Gestures
1. Place gesture templates in `templates/gestures/`
2. Configure mappings in `templates/mappings/`
3. Restart the application

## ⚙️ Configuration

Edit `config/settings.py` to customize:
- Camera resolution and FPS
- Gesture recognition sensitivity
- Keyboard simulation settings
- Logging levels

## 🛠️ Development

### System Checks Performed
- **Compatibility Check**: Python version, required libraries
- **Screen Resolution**: Display configuration validation
- **File Integrity**: Ensures all required files are present
- **Template Validation**: Verifies template folder structure
- **Webcam Access**: Tests camera permissions and functionality

### Adding New Features
1. Follow the modular structure
2. Add new modules to appropriate directories
3. Update imports in `main.py`
4. Update this README accordingly

## 📋 Requirements

See `config/requirements.txt` for detailed dependencies. Key requirements include:
- OpenCV for computer vision
- MediaPipe for hand tracking
- PyAutoGUI/pynput for keyboard simulation
- NumPy for data processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code structure
4. Add appropriate error handling
5. Update documentation
6. Submit a pull request

## 📝 License

[Add your license information here]

## 🐛 Troubleshooting

### Common Issues:
- **Webcam not detected**: Check camera permissions and connections
- **Import errors**: Ensure all dependencies are installed
- **Template errors**: Verify template folder structure and file formats

### Getting Help:
- Check the logs in `utils/logger.py` output
- Ensure all initialization checks pass
- Verify Python version compatibility

---

**Created by:** hurkov  
**Last Updated:** 2025-07-09  
**Version:** 1.0.0
