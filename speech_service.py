import requests
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

class SpeechService:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.api_url = "https://api.deepseek.com/v1/audio/transcriptions"  # 请根据实际API地址修改
        
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """
        将音频数据转换为文本
        :param audio_data: numpy array of audio data
        :param sample_rate: sample rate of the audio
        :return: transcribed text
        """
        try:
            # 将音频数据转换为适合API的格式
            # 这里需要根据Deepseek API的具体要求进行转换
            audio_bytes = self._convert_audio_to_bytes(audio_data, sample_rate)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "audio/wav"  # 根据实际API要求修改
            }
            
            files = {
                "file": ("audio.wav", audio_bytes, "audio/wav")
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                print(f"API请求失败: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"语音识别出错: {str(e)}")
            return ""
            
    def _convert_audio_to_bytes(self, audio_data, sample_rate):
        """
        将numpy数组转换为音频字节
        :param audio_data: numpy array of audio data
        :param sample_rate: sample rate of the audio
        :return: audio bytes
        """
        # 这里需要实现音频格式转换
        # 可以使用scipy.io.wavfile或其他音频处理库
        # 具体实现取决于Deepseek API的要求
        pass 