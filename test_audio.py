import sounddevice as sd
import numpy as np
import time

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    print("录音中...")

try:
    print("开始录音测试...")
    print("可用的音频设备：")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
    
    # 使用默认输入设备
    with sd.InputStream(channels=1, samplerate=16000, callback=audio_callback):
        print("按回车键停止录音...")
        input()
        
except Exception as e:
    print(f"错误: {str(e)}")
    import traceback
    print(traceback.format_exc()) 