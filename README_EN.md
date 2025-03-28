# Voice Transcriber

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

[简体中文](./README.md) | English

A real-time voice transcription tool based on the Whisper model, featuring GPU acceleration and a clean, intuitive graphical interface. Currently supports Windows operating system only.

## ✨ Features

- 🎙️ **Real-time Transcription**: Support real-time audio recording and text transcription
- 🚀 **GPU Acceleration**: Automatic detection and support for NVIDIA (CUDA) and AMD (ROCm) GPU acceleration
- 🌍 **Chinese Optimization**: Optimized transcription performance for Chinese speech
- 🎯 **High Accuracy**: Utilizing OpenAI's Whisper model for accurate transcription
- 🛠️ **Visual Interface**: Modern GUI developed with PySide6
- 📊 **Real-time Feedback**: Live status feedback for audio input

## 🚀 Quick Start

### Requirements

- Windows 10/11 OS
- Python 3.8+
- Microphone device
- NVIDIA GPU (optional, CUDA support) or AMD GPU (optional, ROCm support)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Weedychen/voice_transcriber.git
   cd voice_transcriber
   ```

2. Create and activate virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Launch the application:
   ```bash
   python main.py
   ```

2. In the GUI:
   - Click "Start Recording" button to begin
   - View transcription results in real-time
   - Pause or stop recording at any time
   - Export transcribed text

## 📦 Project Structure

```
voice_transcriber/
├── main.py              # Main program entry
├── speech_service.py    # Speech service module
├── test_audio.py       # Audio test module
├── requirements.txt    # Project dependencies
└── icons/             # Icon resources
```

## ⚙️ Configuration

### Audio Settings
- Sample Rate: 16000Hz
- Audio Format: WAV
- Channel: Mono

### Whisper Model Configuration
- Default model: "tiny"
- Available models: tiny, base, small, medium, large
- GPU acceleration support (requires CUDA or ROCm environment)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [PySide6](https://wiki.qt.io/Qt_for_Python) - GUI framework
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio processing

## 📞 Contact

If you have any questions or suggestions, feel free to:

- Submit an [Issue](https://github.com/Weedychen/voice_transcriber/issues)
- Email: weedychen@outlook.com

## 🔄 Changelog

### [1.0.0] - 2024-03-28
- Initial release
- Real-time voice transcription support
- GPU acceleration support
- Basic GUI implementation 