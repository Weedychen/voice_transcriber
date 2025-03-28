import sys
import json
import numpy as np
import sounddevice as sd
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QTextEdit, QLabel, QComboBox, QFileDialog, QMessageBox, 
                           QProgressBar, QGroupBox, QSlider, QCheckBox, QTabWidget, QSplitter)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QIcon, QColor, QPalette
import requests
from dotenv import load_dotenv
import os
import wave
import io
import base64
import traceback
import logging
from threading import Event, Thread
from queue import Queue, Empty
import time
import whisper
import torch
import urllib.request
import shutil

# 配置日志
logging.basicConfig(
    filename='voice_transcriber.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'  # 添加UTF-8编码
)

# 加载环境变量
load_dotenv()

class ProgressBar:
    def __init__(self, status_signal):
        self.status_signal = status_signal
        self.last_percent = 0
        
    def __call__(self, count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if percent > self.last_percent:
            self.status_signal.emit(f"正在下载Whisper模型... {percent}%")
            self.last_percent = percent

class AudioProcessor(QObject):
    """音频处理类，用于在单独的线程中处理音频数据"""
    text_ready = Signal(str)  # 当有新的转录文本时发出信号
    error_occurred = Signal(str)  # 当发生错误时发出信号
    status_update = Signal(str)  # 状态更新信号
    progress_update = Signal(int)  # 下载进度更新信号
    model_loaded = Signal(str)  # 模型加载完成信号
    download_finished = Signal()  # 下载完成信号
    button_state_update = Signal(bool, str)  # 新增按钮状态更新信号
    model_status_update = Signal(str)  # 新增模型状态更新信号

    def __init__(self):
        super().__init__()
        logging.info("初始化AudioProcessor")
        self.audio_queue = Queue(maxsize=100)
        self.is_processing = False
        self.last_process_time = 0
        self.min_process_interval = 1.5
        self.accumulated_audio = []
        self.sample_rate = 16000
        self.model = None
        self.current_model_name = None
        
        try:
            self.status_update.emit("正在加载Whisper模型...")
            logging.info("正在加载Whisper模型...")
            
            # 使用小型模型以减少加载时间
            model_name = "tiny"
            
            try:
                # 检测可用的GPU设备
                device = self.get_available_device()
                self.model = whisper.load_model(model_name)
                self.model.to(device)
                self.current_model_name = model_name
                logging.info(f"Whisper模型加载完成，使用设备: {device}")
                self.status_update.emit(f"Whisper模型加载完成，使用设备: {device}")
            except Exception as e:
                error_msg = f"加载模型失败: {str(e)}"
                logging.error(error_msg)
                logging.error(traceback.format_exc())
                self.error_occurred.emit(error_msg)
                raise
                
        except Exception as e:
            error_msg = f"初始化音频处理器失败: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            self.error_occurred.emit(error_msg)
            raise

    def process_audio(self, audio_data):
        try:
            if not isinstance(audio_data, (list, np.ndarray)) or len(audio_data) == 0:
                return

            # 检查音频数据是否有声音
            audio_array = np.array(audio_data)
            rms = np.sqrt(np.mean(np.square(audio_array)))
            
            if rms < 0.003:
                return

            # 累积音频数据
            self.accumulated_audio.extend(audio_array)
            
            # 如果累积的数据超过2秒，进行处理
            if len(self.accumulated_audio) >= self.sample_rate * 2:
                current_time = time.time()
                if current_time - self.last_process_time < self.min_process_interval:
                    return

                try:
                    self.status_update.emit("正在转录音频...")
                    logging.info("开始音频转录...")
                    
                    # 准备音频数据
                    audio_data = np.array(self.accumulated_audio[-self.sample_rate * 2:])
                    audio_data = audio_data.astype(np.float32)
                    
                    # 确保音频数据是单声道
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data[:, 0]
                    
                    # 归一化音频
                    if np.abs(audio_data).max() > 0:
                        audio_data = audio_data / np.abs(audio_data).max()
                    
                    # 检查是否可以使用半精度浮点数(fp16)
                    use_fp16 = torch.cuda.is_available() or (hasattr(torch, 'hip') and torch.hip.is_available())
                    
                    # 使用更多的转录选项
                    result = self.model.transcribe(
                        audio_data,
                        language="zh",
                        task="transcribe",
                        fp16=use_fp16,
                        best_of=3,
                        beam_size=3
                    )
                    
                    text = result["text"].strip()
                    if text:
                        logging.info(f"转录结果: {text}")
                        self.text_ready.emit(text)
                        self.status_update.emit("转录完成")
                    
                    self.last_process_time = current_time
                    # 保留最后0.5秒的数据，以保持连续性
                    self.accumulated_audio = self.accumulated_audio[-int(self.sample_rate * 0.5):]
                        
                except Exception as e:
                    error_msg = f"转录出错: {str(e)}"
                    self.error_occurred.emit(error_msg)
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                
        except Exception as e:
            error_msg = f"处理音频时出错: {str(e)}"
            self.error_occurred.emit(error_msg)
            logging.error(error_msg)
            logging.error(traceback.format_exc())

    def get_available_device(self):
        """检测并返回可用的设备（CUDA、ROCm或CPU）"""
        device = "cpu"
        device_info = "CPU"
        
        try:
            # 检测ROCm支持（AMD显卡）
            if hasattr(torch, 'hip') and torch.hip.is_available():
                device = "cuda"  # PyTorch中ROCm也使用cuda作为设备名
                device_info = f"ROCm (AMD GPU: {torch.cuda.get_device_name(0)})"
                logging.info(f"检测到AMD GPU，使用ROCm后端")
            # 检测CUDA支持（NVIDIA显卡）
            elif torch.cuda.is_available():
                device = "cuda"
                device_info = f"CUDA (NVIDIA GPU: {torch.cuda.get_device_name(0)})"
                logging.info(f"检测到NVIDIA GPU，使用CUDA后端")
            else:
                logging.info("未检测到GPU，使用CPU模式")
            
            # 验证设备是否真正可用
            if device != "cpu":
                try:
                    # 尝试分配一个小张量到GPU
                    test_tensor = torch.zeros((1,), device=device)
                    del test_tensor
                except RuntimeError as e:
                    logging.warning(f"GPU设备测试失败: {str(e)}")
                    device = "cpu"
                    device_info = "CPU (GPU测试失败，已回退到CPU模式)"
        except Exception as e:
            logging.error(f"设备检测错误: {str(e)}")
            device = "cpu"
            device_info = "CPU (设备检测出错，使用CPU模式)"
        
        logging.info(f"检测到可用设备: {device_info}")
        return device
        
    def load_model(self, model_name):
        """加载指定的模型"""
        try:
            self.status_update.emit(f"正在加载{model_name}模型...")
            self.model_status_update.emit(f"模型状态: 正在加载{model_name}模型...")
            logging.info(f"正在加载{model_name}模型...")
            
            # 发送初始进度信号
            self.progress_update.emit(10)
            
            # 检测可用的GPU设备
            device = self.get_available_device()
            self.progress_update.emit(30)
            
            # 加载模型 - 这是最耗时的部分
            self.model = whisper.load_model(model_name)
            self.progress_update.emit(70)
            
            # 将模型移动到设备
            self.model.to(device)
            self.current_model_name = model_name
            self.progress_update.emit(90)
            
            success_msg = f"Whisper模型{model_name}加载完成，使用设备: {device}"
            logging.info(success_msg)
            self.status_update.emit(success_msg)
            self.model_status_update.emit(f"模型状态: {model_name}模型已加载")
            self.model_loaded.emit(model_name)
            self.button_state_update.emit(True, "下载选中的模型")
            self.progress_update.emit(100)
            
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            self.error_occurred.emit(error_msg)
            self.button_state_update.emit(True, "下载选中的模型")
            self.model_status_update.emit("模型状态: 加载失败")
            raise

class VoiceTranscriber(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音转录工具")
        self.setGeometry(100, 100, 900, 700)

        # 设置日志
        logging.basicConfig(
            filename='voice_transcriber.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )

        # 设置应用样式
        self.setStyleSheet("")

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 创建顶部区域（模型和设备设置）
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setSpacing(15)

        # 创建模型选择区域
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(8)

        # 模型选择下拉框
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("选择Whisper模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "tiny (适合快速测试，39M)",
            "base (基础模型，142M)",
            "small (准确度更高，466M)",
            "medium (高准确度，1.5G)",
            "large (最高准确度，3G)"
        ])
        model_select_layout.addWidget(self.model_combo)
        model_layout.addLayout(model_select_layout)

        # 模型下载按钮
        self.download_button = QPushButton("下载选中的模型")
        self.download_button.clicked.connect(self.download_model)
        model_layout.addWidget(self.download_button)

        # 模型状态标签
        self.model_status = QLabel("模型状态: 未加载")
        model_layout.addWidget(self.model_status)

        # 创建硬件信息区域
        hardware_layout = QHBoxLayout()
        # 检测GPU信息
        gpu_info = "未检测到GPU"
        if torch.cuda.is_available():
            gpu_info = f"NVIDIA GPU: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            gpu_info = "AMD GPU (ROCm)"
        hardware_layout.addWidget(QLabel(f"硬件加速: {gpu_info}"))
        model_layout.addLayout(hardware_layout)

        # 创建设备选择区域
        device_group = QGroupBox("录音设置")
        device_layout = QVBoxLayout(device_group)
        device_layout.setSpacing(8)

        # 麦克风选择
        device_select_layout = QHBoxLayout()
        device_select_layout.addWidget(QLabel("选择麦克风:"))
        self.device_combo = QComboBox()
        self.update_device_list()
        device_select_layout.addWidget(self.device_combo)
        device_layout.addLayout(device_select_layout)

        # 说话人选择
        speaker_select_layout = QHBoxLayout()
        speaker_select_layout.addWidget(QLabel("当前说话人:"))
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems(["说话人1", "说话人2", "说话人3"])
        speaker_select_layout.addWidget(self.speaker_combo)
        device_layout.addLayout(speaker_select_layout)

        # 添加到顶部布局
        top_layout.addWidget(model_group, 1)
        top_layout.addWidget(device_group, 1)
        main_layout.addWidget(top_widget)

        # 创建中间区域（状态和控制）
        middle_widget = QWidget()
        middle_layout = QHBoxLayout(middle_widget)
        middle_layout.setSpacing(10)

        # 状态区域
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout(status_group)

        # 音量显示
        self.volume_label = QLabel("音量: 0 dB")
        status_layout.addWidget(self.volume_label)

        # 状态标签
        self.status_label = QLabel("状态: 正在初始化...")
        status_layout.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)  # 初始时隐藏进度条
        status_layout.addWidget(self.progress_bar)

        # 控制区域
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout(control_group)

        # 录音按钮
        self.record_button = QPushButton("开始录音")
        self.record_button.setMinimumHeight(40)
        self.record_button.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_button)

        # 导出按钮
        self.export_button = QPushButton("导出对话记录")
        self.export_button.clicked.connect(self.export_transcript)
        control_layout.addWidget(self.export_button)

        # 添加到中间布局
        middle_layout.addWidget(status_group, 2)
        middle_layout.addWidget(control_group, 1)
        main_layout.addWidget(middle_widget)

        # 创建文本显示区域
        transcript_group = QGroupBox("对话记录")
        transcript_layout = QVBoxLayout(transcript_group)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMinimumHeight(300)
        transcript_layout.addWidget(self.text_display)

        main_layout.addWidget(transcript_group, 1)
        
        # 初始化录音状态和对话记录
        self.is_recording = False
        self.audio_queue = Queue()
        self.stop_event = Event()
        self.conversation_history = []
        self.last_speaker = None
        self.silence_start = None
        self.last_voice_time = time.time()
        
        # 创建音频处理器
        self.audio_processor = AudioProcessor()
        self.audio_processor.text_ready.connect(self.handle_transcribed_text)
        self.audio_processor.error_occurred.connect(self.handle_error)
        self.audio_processor.status_update.connect(self.handle_status_update)
        self.audio_processor.progress_update.connect(self.update_progress)
        self.audio_processor.model_loaded.connect(self.handle_model_loaded)
        self.audio_processor.button_state_update.connect(self.update_button_state)
        self.audio_processor.model_status_update.connect(self.update_model_status)
        
        # 创建音频处理线程
        self.process_thread = Thread(target=self.process_audio_queue, daemon=True)
        self.process_thread.start()

        # 创建自动说话人检测定时器
        self.voice_timer = QTimer()
        self.voice_timer.timeout.connect(self.check_voice_activity)
        self.voice_timer.start(500)  # 每500ms检查一次

        # 修改模型选择下拉框连接
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        
        # 添加下载状态变量
        self.is_downloading = False

    def show_error(self, message):
        """显示错误消息"""
        logging.error(message)
        QMessageBox.critical(self, "错误", message)
        
    def handle_error(self, error_message):
        """处理错误信号"""
        self.status_label.setText(f"错误: {error_message}")
        logging.error(error_message)
        
    def handle_transcribed_text(self, text):
        """处理转录文本信号"""
        current_speaker = self.speaker_combo.currentText()
        
        # 记录对话
        self.conversation_history.append({
            "speaker": current_speaker,
            "text": text,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # 更新显示
        self.update_conversation_display()

    def update_conversation_display(self):
        """更新对话显示"""
        self.text_display.clear()
        for entry in self.conversation_history:
            speaker = entry["speaker"]
            text = entry["text"]
            timestamp = entry["timestamp"]
            
            # 使用HTML格式化显示
            html = f'<p><b>{speaker}</b> <span style="color: gray;">({timestamp})</span><br>{text}</p>'
            self.text_display.append(html)
        
        # 滚动到底部
        self.text_display.verticalScrollBar().setValue(
            self.text_display.verticalScrollBar().maximum()
        )

    def export_transcript(self):
        """导出对话记录"""
        try:
            if not self.conversation_history:
                self.show_error("没有可导出的对话记录")
                return
                
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "保存对话记录",
                f"对话记录_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Text Files (*.txt)"
            )
            
            if file_name:
                with open(file_name, 'w', encoding='utf-8') as f:
                    for entry in self.conversation_history:
                        f.write(f"[{entry['timestamp']}] {entry['speaker']}:\n")
                        f.write(f"{entry['text']}\n\n")
                logging.info(f"对话记录已保存到: {file_name}")
                self.status_label.setText("状态: 对话记录已导出")
        except Exception as e:
            self.show_error(f"导出记录失败: {str(e)}")

    def handle_status_update(self, status):
        """处理状态更新信号"""
        self.status_label.setText(f"状态: {status}")
        
        # 处理进度条显示
        if "正在下载Whisper模型..." in status:
            self.progress_bar.setVisible(True)
            try:
                # 从状态消息中提取进度百分比
                percent = int(status.split("...")[-1].strip().replace("%", ""))
                self.progress_bar.setValue(percent)
            except:
                pass
        elif "Whisper模型下载完成" in status:
            self.progress_bar.setVisible(False)
        elif "Whisper模型加载完成" in status:
            self.progress_bar.setVisible(False)
        
    def update_device_list(self):
        """更新音频设备列表"""
        try:
            devices = sd.query_devices()
            self.device_combo.clear()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # 只显示输入设备
                    self.device_combo.addItem(f"{i}: {device['name']}", i)
            logging.info("音频设备列表已更新")
        except Exception as e:
            self.show_error(f"获取音频设备列表失败: {str(e)}")
        
    def toggle_recording(self):
        try:
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_recording()
        except Exception as e:
            self.show_error(f"录音操作失败: {str(e)}")
            
    def start_recording(self):
        try:
            # 检查是否已加载模型
            if not hasattr(self.audio_processor, 'model'):
                raise ValueError("请先下载并加载模型")
                
            # 获取选中的设备ID
            device_id = self.device_combo.currentData()
            if device_id is None:
                raise ValueError("请选择音频设备")
            
            # 获取设备信息
            device_info = sd.query_devices(device_id)
            logging.info(f"使用设备: {device_info}")
            
            # 重置状态
            self.is_recording = True
            # 清空队列
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except Empty:
                    break
            self.record_button.setText("停止录音")
            self.status_label.setText("正在录音...")
            
            logging.info(f"开始录音，使用设备ID: {device_id}")
            
            # 使用选中的输入设备
            self.stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=16000,
                callback=self.audio_callback,
                blocksize=1024,  # 减小缓冲区大小以降低延迟
                dtype=np.float32,
                latency='low'  # 使用低延迟模式
            )
            self.stream.start()
            
        except Exception as e:
            self.show_error(f"启动录音失败: {str(e)}")
            self.stop_recording()
        
    def stop_recording(self):
        try:
            self.is_recording = False
            self.record_button.setText("开始录音")
            self.status_label.setText("就绪")
            self.volume_label.setText("音量: 0 dB")
            
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            logging.info("停止录音")
        except Exception as e:
            self.show_error(f"停止录音失败: {str(e)}")
        
    def audio_callback(self, indata, frames, time_info, status):
        try:
            if status:
                logging.warning(f"音频回调状态: {status}")
            
            # 计算音量
            volume_norm = np.linalg.norm(indata) * 10
            volume_db = 20 * np.log10(volume_norm) if volume_norm > 0 else -60
            self.volume_label.setText(f"音量: {volume_db:.1f} dB")
            
            # 更新最后检测到声音的时间
            if volume_db > -30:  # 声音阈值
                self.last_voice_time = time.time()
            
            # 记录音频数据
            audio_chunk = indata[:, 0].copy()
            
            # 检查音频数据
            if np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
                logging.warning("检测到无效的音频数据")
                return
                
            # 如果队列太大，清空它
            if self.audio_queue.qsize() >= 16000 * 3:  # 限制缓冲区大小为3秒
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except Empty:
                        break
            
            self.audio_queue.put(audio_chunk)
                
        except Exception as e:
            logging.error(f"音频回调错误: {str(e)}")
            logging.error(traceback.format_exc())

    def check_voice_activity(self):
        """检查语音活动，自动识别说话人"""
        if self.is_recording:
            current_time = time.time()
            if current_time - self.last_voice_time > 2.0:  # 2秒无声判定为说话人切换
                if self.silence_start is None:
                    self.silence_start = current_time
                elif current_time - self.silence_start > 2.0:
                    # 分析音频特征识别说话人
                    if len(self.accumulated_audio) > 0:
                        audio_features = self.extract_voice_features(self.accumulated_audio)
                        speaker_id = self.identify_speaker(audio_features)
                        if speaker_id >= 0:
                            self.speaker_combo.setCurrentIndex(speaker_id)
                    self.silence_start = None
            else:
                self.silence_start = None
                
    def extract_voice_features(self, audio_data):
        """提取语音特征用于说话人识别"""
        # 这里可以添加MFCC等声纹特征提取逻辑
        return {
            'rms': np.sqrt(np.mean(np.square(audio_data))),
            'spectral_centroid': np.mean(np.abs(np.fft.fft(audio_data)))
        }
        
    def identify_speaker(self, features):
        """根据语音特征识别说话人"""
        # 这里可以添加说话人识别逻辑
        # 简单实现：根据特征匹配预存的说话人特征
        return 0  # 暂时返回第一个说话人

    def process_audio_queue(self):
        while not self.stop_event.is_set():
            try:
                audio_chunk = self.audio_queue.get()
                self.audio_processor.process_audio(audio_chunk)
            except Empty:
                time.sleep(0.01)  # 等待音频数据
            except Exception as e:
                logging.error(f"音频处理错误: {str(e)}")
                logging.error(traceback.format_exc())

    def update_progress(self, progress):
        """更新进度条"""
        if not self.is_downloading:
            return
            
        self.progress_bar.setValue(progress)
        self.progress_bar.setVisible(True)
        logging.debug(f"进度条更新: {progress}%")
        
    def handle_model_loaded(self, model_name):
        """处理模型加载完成信号"""
        self.is_downloading = False
        QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))

    def on_model_selected(self, index):
        """当选择新模型时触发"""
        if not self.is_downloading and self.audio_processor.current_model_name:
            model_text = self.model_combo.currentText()
            new_model = model_text.split()[0]
            if new_model != self.audio_processor.current_model_name:
                reply = QMessageBox.question(
                    self,
                    "切换模型",
                    f"是否要切换到{new_model}模型？\n注意：切换模型会暂停当前录音。",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.download_model()

    def download_model(self):
        """下载选中的Whisper模型"""
        try:
            if self.is_downloading:
                return
                
            # 获取选中的模型名称
            model_text = self.model_combo.currentText()
            model_name = model_text.split()[0]
            
            # 如果正在录音，先停止录音
            if self.is_recording:
                self.stop_recording()
            
            # 设置下载状态
            self.is_downloading = True
            self.download_button.setEnabled(False)
            self.download_button.setText("正在下载模型...")
            self.model_status.setText("模型状态: 正在下载...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # 记录下载状态到日志
            logging.debug(f"开始下载模型: {model_name}, is_downloading={self.is_downloading}")
            
            # 在新线程中加载模型
            load_thread = Thread(target=self._load_model_thread, args=(model_name,))
            load_thread.daemon = True
            load_thread.start()
            
        except Exception as e:
            self.is_downloading = False
            self.show_error(f"启动模型下载失败: {str(e)}")
            self.download_button.setEnabled(True)
            self.download_button.setText("下载选中的模型")
            
    def _load_model_thread(self, model_name):
        """在后台线程中加载模型"""
        try:
            self.audio_processor.load_model(model_name)
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            self.audio_processor.error_occurred.emit(error_msg)
        finally:
            self.is_downloading = False

    def update_button_state(self, enabled, text):
        """更新按钮状态"""
        self.download_button.setEnabled(enabled)
        self.download_button.setText(text)
        
    def update_model_status(self, status):
        """更新模型状态"""
        self.model_status.setText(status)

if __name__ == '__main__':
    try:
        logging.info("程序启动")
        app = QApplication(sys.argv)
        window = VoiceTranscriber()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"程序启动失败: {str(e)}")
        logging.error(traceback.format_exc())