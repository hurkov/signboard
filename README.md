# signboard

sign_language_keyboard/
├── main.py                    # Your main entry point
├── config/
│   ├── __init__.py
│   ├── settings.py           # Configuration constants
│   └── requirements.txt      # Dependencies
├── core/
│   ├── __init__.py
│   ├── compatibility_checker.py  # Your check #1
│   ├── system_checker.py         # Your checks #2, #3, #5
│   └── template_manager.py       # Your check #4 + template operations
├── input/
│   ├── __init__.py
│   ├── camera_handler.py         # Webcam operations
│   └── gesture_detector.py       # Hand tracking & recognition
├── output/
│   ├── __init__.py
│   └── keyboard_simulator.py     # Your keyboard typing module
├── templates/                    # Your templates folder
│   ├── gestures/
│   └── mappings/
└── utils/
    ├── __init__.py
    └── logger.py                 # Error handling & logging
