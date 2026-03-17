# -*- coding: utf-8 -*-
"""
阿里云百炼语音合成(TTS)服务模块
负责将文本转换为语音并播放
"""

import os
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback
from datetime import datetime
from pathlib import Path
from config import *


def get_timestamp():
    """获取当前时间戳"""
    now = datetime.now()
    formatted_timestamp = now.strftime("[%Y-%m-%d %H:%M:%S.%f]")
    return formatted_timestamp


class TTSCallback(ResultCallback):
    """TTS回调处理类"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.file = None
        self.is_complete = False  # 标记是否完成
        self.error_message = None  # 错误信息
    
    def on_open(self):
        """连接建立时调用"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            # 打开输出文件，准备写入音频数据
            self.file = open(self.output_path, "wb")
            print(f"🔊 TTS连接建立：{get_timestamp()}")
        except Exception as e:
            print(f"❌ 打开音频文件失败: {e}")
            self.error_message = str(e)
    
    def on_complete(self):
        """语音合成完成时调用"""
        print(f"✅ 语音合成完成：{get_timestamp()}")
        if self.file:
            self.file.close()
            self.file = None
        
        # 打印性能指标
        if hasattr(self, 'synthesizer'):
            print(f'[性能] requestId: {self.synthesizer.get_last_request_id()}，'
                  f'首包延迟: {self.synthesizer.get_first_package_delay()}毫秒')
        
        # 标记完成
        self.is_complete = True
    
    def on_error(self, message: str):
        """发生错误时调用"""
        print(f"❌ 语音合成出现异常：{message}")
        self.error_message = message
        if self.file:
            self.file.close()
            self.file = None
        self.is_complete = True  # 出错也算完成
    
    def on_close(self):
        """连接关闭时调用"""
        print(f"🔌 TTS连接关闭：{get_timestamp()}")
        if self.file:
            self.file.close()
            self.file = None
    
    def on_event(self, message):
        """事件回调"""
        pass
    
    def on_data(self, data: bytes) -> None:
        """接收音频数据时调用"""
        if self.file:
            self.file.write(data)
            # 只在第一次接收数据时打印
            if not hasattr(self, '_first_data_logged'):
                print(f"{get_timestamp()} 开始接收音频数据...")
                self._first_data_logged = True


class TTSService:
    """语音合成服务类"""
    
    def __init__(self):
        """初始化TTS服务"""
        # 设置API Key
        dashscope.api_key = ALIBABA_API_KEY
        
        # 配置参数
        self.model = TTS_MODEL
        self.voice = TTS_VOICE
        self.rate = TTS_RATE
        self.output_dir = TTS_OUTPUT_DIR
        self.enabled = TTS_ENABLED
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.enabled:
            print(f"🎤 TTS服务初始化完成")
            print(f"   模型: {self.model}")
            print(f"   音色: {self.voice}")
            print(f"   语速: {self.rate}")
        else:
            print("🔇 TTS服务已禁用")
    
    def synthesize_and_play(self, text: str, play_audio: bool = True, async_play: bool = False) -> str:
        """
        合成语音并播放
        
        Args:
            text: 要合成的文本
            play_audio: 是否播放音频（True=播放，False=仅生成文件）
            async_play: 是否异步播放（True=开始播放后立即返回，False=等待播放完成）
        
        Returns:
            生成的音频文件路径，失败返回None
        """
        if not self.enabled:
            print("⚠️ TTS服务未启用，跳过语音合成")
            return None
        
        if not text or len(text.strip()) == 0:
            print("⚠️ 文本为空，跳过语音合成")
            return None
        
        try:
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"tts_{timestamp}.mp3")
            
            print(f"🎵 开始语音合成...")
            print(f"   文本长度: {len(text)} 字符")
            print(f"   输出路径: {output_path}")
            
            # 创建回调对象
            callback = TTSCallback(output_path)
            
            # 实例化SpeechSynthesizer
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                callback=callback,
            )
            
            # 将synthesizer实例赋值给callback，以便在回调中使用
            callback.synthesizer = synthesizer
            
            # 构建SSML文本（支持语速控制）
            # 注意：只有特定模型和音色支持SSML
            ssml_text = f'<speak rate="{self.rate}">{self._escape_xml(text)}</speak>'
            
            # 调用语音合成（单向流式）
            synthesizer.call(ssml_text)
            
            # 等待合成完成（轮询callback的is_complete标志）
            import time
            max_wait_time = 30  # 最多等待30秒
            wait_interval = 0.1  # 每次检查间隔0.1秒
            elapsed_time = 0
            
            while not callback.is_complete and elapsed_time < max_wait_time:
                time.sleep(wait_interval)
                elapsed_time += wait_interval
            
            # 检查是否超时
            if not callback.is_complete:
                print(f"⚠️ 语音合成超时（等待了{max_wait_time}秒）")
                return None
            
            # 检查是否有错误
            if callback.error_message:
                print(f"❌ 语音合成失败: {callback.error_message}")
                return None
            
            # 确保文件句柄已关闭
            if callback.file:
                try:
                    callback.file.close()
                    callback.file = None
                except:
                    pass
            
            # 额外等待确保文件完全写入磁盘
            time.sleep(1)
            
            # 规范化路径（处理反斜杠）
            normalized_path = os.path.abspath(output_path)
            
            # 检查文件是否生成成功
            if os.path.exists(normalized_path):
                file_size = os.path.getsize(normalized_path)
                if file_size > 0:
                    print(f"✅ 语音文件生成成功: {normalized_path}")
                    print(f"   文件大小: {file_size} 字节")
                    
                    # 播放音频（如果需要）
                    if play_audio:
                        if async_play:
                            # 异步播放：在新线程中播放，立即返回
                            import threading
                            play_thread = threading.Thread(
                                target=self._play_audio, 
                                args=(normalized_path,),
                                daemon=True
                            )
                            play_thread.start()
                            print("🎵 音频开始异步播放...")
                        else:
                            # 同步播放：等待播放完成
                            self._play_audio(normalized_path)
                    
                    return normalized_path
                else:
                    print(f"❌ 语音文件为空（大小: {file_size} 字节）")
                    return None
            else:
                print(f"❌ 语音文件不存在: {normalized_path}")
                # 尝试检查原始路径
                if os.path.exists(output_path):
                    print(f"   但原始路径存在: {output_path}")
                return None
                
        except Exception as e:
            print(f"❌ 语音合成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _escape_xml(self, text: str) -> str:
        """转义XML特殊字符"""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace("\"", "&quot;")
        text = text.replace("'", "&apos;")
        return text
    
    def _play_audio(self, audio_path: str):
        """
        播放音频文件
        
        Args:
            audio_path: 音频文件路径
        """
        try:
            print(f"🔊 播放音频: {audio_path}")
            
            # 根据操作系统选择播放方式
            import platform
            system = platform.system()
            
            if system == "Windows":
                # Windows系统优先使用pygame（支持MP3），其次尝试playsound，最后使用系统命令
                try:
                    # 方案1: 使用pygame播放（最可靠，支持MP3）
                    import pygame
                    
                    # 确保路径存在且可访问
                    if not os.path.exists(audio_path):
                        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
                    
                    # 规范化路径
                    audio_path = os.path.abspath(audio_path)
                    print(f"   使用pygame播放，路径: {audio_path}")
                    
                    pygame.mixer.init()
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    
                    # 等待播放完成
                    import time
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    pygame.mixer.quit()
                    print("✅ 音频播放完成（pygame）")
                    
                except ImportError:
                    # 方案2: 尝试使用playsound
                    try:
                        from playsound import playsound
                        playsound(audio_path)
                        print("✅ 音频播放完成（playsound）")
                        
                    except (ImportError, Exception) as e:
                        # 方案3: 使用Windows系统命令播放
                        print(f"⚠️ pygame和playsound不可用，尝试使用系统命令")
                        import subprocess
                        
                        # 使用start命令打开默认播放器
                        subprocess.Popen(['start', '', audio_path], shell=True)
                        print("✅ 已使用系统默认播放器打开音频")
                        print("   提示: 推荐安装pygame获得更好体验: pip install pygame")
            
            elif system == "Darwin":  # macOS
                import subprocess
                subprocess.call(["afplay", audio_path])
                print("✅ 音频播放完成")
            
            elif system == "Linux":
                import subprocess
                # 尝试使用多个Linux音频播放器
                players = ["mpg123", "ffplay", "aplay"]
                for player in players:
                    try:
                        subprocess.call([player, audio_path])
                        print("✅ 音频播放完成")
                        return
                    except FileNotFoundError:
                        continue
                print(f"⚠️ 未找到音频播放器，请安装: sudo apt-get install mpg123")
                print(f"   音频文件已保存: {audio_path}")
            
            else:
                print(f"⚠️ 不支持的操作系统: {system}")
                print(f"   音频文件已保存: {audio_path}")
                
        except Exception as e:
            print(f"⚠️ 播放音频时出错: {e}")
            import traceback
            traceback.print_exc()
            print(f"   音频文件已保存: {audio_path}")
    
    def clean_old_audio_files(self, keep_last_n: int = 10):
        """
        清理旧的音频文件，只保留最近的N个
        
        Args:
            keep_last_n: 保留的文件数量
        """
        try:
            audio_files = []
            for file in os.listdir(self.output_dir):
                if file.startswith("tts_") and file.endswith(".mp3"):
                    file_path = os.path.join(self.output_dir, file)
                    audio_files.append((file_path, os.path.getmtime(file_path)))
            
            # 按修改时间排序
            audio_files.sort(key=lambda x: x[1], reverse=True)
            
            # 删除旧文件
            for file_path, _ in audio_files[keep_last_n:]:
                os.remove(file_path)
                print(f"🗑️  删除旧音频文件: {file_path}")
                
        except Exception as e:
            print(f"⚠️ 清理音频文件时出错: {e}")


# 测试代码
if __name__ == "__main__":
    tts = TTSService()
    
    test_text = "你好，我是你的心理咨询伴侣。我会用温暖的声音陪伴你，倾听你的心声。"
    print(f"测试文本: {test_text}")
    
    audio_path = tts.synthesize_and_play(test_text, play_audio=True)
    
    if audio_path:
        print(f"\n✅ 测试成功！音频文件: {audio_path}")
    else:
        print(f"\n❌ 测试失败")
