# Voice Transcriber (语音转写工具)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

[English](./README_EN.md) | 简体中文

基于 Whisper 模型的实时语音转写工具，支持GPU加速，提供简洁直观的图形界面。本工具目前仅支持Windows操作系统。

## ✨ 功能特点

- 🎙️ **实时语音转写**：支持实时录音并转写为文字
- 🚀 **GPU加速支持**：自动检测并支持NVIDIA（CUDA）和AMD（ROCm）显卡加速
- 🌍 **中文优化**：针对中文语音优化的转写效果
- 🎯 **高精度转写**：使用OpenAI的Whisper模型确保转写准确性
- 🛠️ **可视化界面**：基于PySide6开发的现代化图形界面
- 📊 **实时反馈**：提供音频输入的实时状态反馈

## 🚀 快速开始

### 环境要求

- Windows 10/11 操作系统
- Python 3.8+
- 麦克风设备
- NVIDIA GPU（可选，支持CUDA）或 AMD GPU（可选，支持ROCm）

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/Weedychen/voice_transcriber.git
   cd voice_transcriber
   ```

2. 创建并激活虚拟环境（推荐）：
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 使用方法

1. 启动应用：
   ```bash
   python main.py
   ```

2. 在图形界面中：
   - 点击"开始录音"按钮开始录制
   - 实时查看转写结果
   - 可随时暂停或停止录制
   - 支持导出转写文本

## 📦 项目结构

```
voice_transcriber/
├── main.py              # 主程序入口
├── speech_service.py    # 语音服务模块
├── test_audio.py       # 音频测试模块
├── requirements.txt    # 项目依赖
└── icons/             # 图标资源目录
```

## ⚙️ 配置说明

### 音频设置
- 采样率：16000Hz
- 音频格式：WAV
- 声道：单声道

### Whisper模型配置
- 默认使用"tiny"模型
- 可选模型：tiny, base, small, medium, large
- 支持GPU加速（需要CUDA或ROCm环境）

## 🤝 贡献指南

欢迎提交问题和改进建议！提交代码请遵循以下步骤：

1. Fork 本仓库
2. 创建您的特性分支 (git checkout -b feature/AmazingFeature)
3. 提交您的更改 (git commit -m 'Add some AmazingFeature')
4. 推送到分支 (git push origin feature/AmazingFeature)
5. 打开一个 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别模型
- [PySide6](https://wiki.qt.io/Qt_for_Python) - GUI框架
- [sounddevice](https://python-sounddevice.readthedocs.io/) - 音频处理

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 [Issue](https://github.com/Weedychen/voice_transcriber/issues)

## 🔄 更新日志

### [1.0.0] - 2024-03-28
- 首次发布
- 支持实时语音转写
- 添加GPU加速支持
- 实现基础GUI界面 