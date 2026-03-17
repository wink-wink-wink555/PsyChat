# 语音合成(TTS)功能使用指南

## 📖 功能简介

本项目集成了阿里云百炼的语音合成(TTS)服务，可以在大模型生成心理咨询回答后，立即将文本转换为温暖的语音播放，提供更加人性化的交互体验。

## ✨ 功能特性

- 🎤 **自动语音合成**：大模型生成回答后立即转换为语音
- 🔊 **自动播放**：合成完成后自动播放音频
- 💾 **音频保存**：所有生成的音频文件都会保存到本地
- 🧹 **自动清理**：自动清理旧音频文件，只保留最近10个
- 🎚️ **可配置**：支持配置音色、语速等参数

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install dashscope playsound
```

或者安装完整依赖：

```bash
pip install -r requirements.txt
```

### 2. 配置参数

在 `config.py` 中已经配置好了TTS参数：

```python
# 阿里云百炼语音合成(TTS)配置
TTS_ENABLED = True  # 是否启用语音合成功能
TTS_MODEL = "cosyvoice-v3-flash"  # 语音合成模型
TTS_VOICE = "longanyang"  # 音色选择
TTS_RATE = "1"  # 语速（0.5-2.0，1为正常语速）
TTS_OUTPUT_DIR = str(PROJECT_ROOT / "storage" / "audio")  # 音频文件输出目录
```

### 3. 正常使用

启动Web界面或命令行模式，TTS功能会自动工作：

```bash
# Web界面模式
python main.py

# 命令行模式
python main.py --cli
```

## ⚙️ 配置说明

### 启用/禁用TTS

在 `config.py` 中修改：

```python
TTS_ENABLED = True  # 启用TTS
TTS_ENABLED = False  # 禁用TTS
```

### 更改音色

支持的音色包括：
- `longanyang` - 温柔男声（默认，适合心理咨询）
- `longxiaochun` - 温柔女声
- `longjing` - 平和女声
- `longxuanxuan` - 亲切女声

修改方式：

```python
TTS_VOICE = "longxiaochun"  # 更改为其他音色
```

### 调整语速

语速范围：0.5 - 2.0
- `0.5` - 最慢
- `1.0` - 正常（默认）
- `2.0` - 最快

```python
TTS_RATE = "1.5"  # 更改为1.5倍速
```

### 更改模型

支持的模型：
- `cosyvoice-v3-flash` - 快速模型（默认）
- `cosyvoice-v3-plus` - 高质量模型
- `cosyvoice-v2` - 经典模型

```python
TTS_MODEL = "cosyvoice-v3-plus"  # 更改为高质量模型
```

## 📁 音频文件管理

### 存储位置

所有音频文件存储在：`storage/audio/`

文件命名格式：`tts_YYYYMMDD_HHMMSS.mp3`

### 自动清理

系统会自动清理旧音频文件，只保留最近10个。

手动清理：

```python
from core.tts_service import TTSService

tts = TTSService()
tts.clean_old_audio_files(keep_last_n=5)  # 只保留最近5个
```

### 手动管理

如需手动删除所有音频文件：

```bash
# Windows
del /q storage\audio\*.mp3

# Linux/macOS
rm storage/audio/*.mp3
```

## 🔧 高级使用

### 仅生成音频不播放

```python
from core.tts_service import TTSService

tts = TTSService()
audio_path = tts.synthesize_and_play(
    text="你好，这是测试文本",
    play_audio=False  # 不播放，只生成文件
)
```

### 在代码中直接使用

```python
from core.tts_service import TTSService

# 初始化服务
tts = TTSService()

# 合成并播放
text = "我理解你的感受，让我们一起来探讨一下。"
audio_path = tts.synthesize_and_play(text, play_audio=True)

if audio_path:
    print(f"音频文件已保存: {audio_path}")
```

## 🐛 故障排除

### 1. 播放失败

**Windows系统**：
```bash
pip install playsound==1.2.2
```

如果还是失败，音频文件仍会保存到 `storage/audio/` 目录，可以手动播放。

**Linux系统**：
```bash
sudo apt-get install mpg123
```

**macOS系统**：
自带 `afplay` 命令，无需额外安装。

### 2. API调用失败

检查API Key是否正确：
```python
# 在 config.py 中
ALIBABA_API_KEY = "sk-your-api-key-here"
```

确保API Key有TTS服务的权限。

### 3. 音频文件为空

可能原因：
1. 网络问题，重试即可
2. 文本包含特殊字符，系统会自动转义
3. API配额不足，检查阿里云控制台

### 4. 延迟较高

- 首次调用会建立WebSocket连接，有一定延迟
- 后续调用会更快
- 可以考虑使用 `cosyvoice-v3-flash` 快速模型

## 📊 性能指标

- **首包延迟**：通常在200-500ms
- **合成速度**：实时合成，流式输出
- **音频质量**：16kHz/24kHz采样率，MP3格式
- **文本长度限制**：建议单次不超过500字

## 🔐 安全说明

- API Key存储在 `config.py` 中，请勿提交到公共仓库
- 建议使用环境变量或 `.env` 文件管理敏感信息
- 音频文件本地存储，不会上传到服务器

## 📝 技术实现

### 架构设计

```
RAG System (生成回答)
    ↓
TTS Service (文本转语音)
    ↓
阿里云百炼 TTS API (语音合成)
    ↓
音频文件保存 (storage/audio/)
    ↓
自动播放 (winsound/playsound/afplay)
```

### 关键模块

1. **TTSService** (`core/tts_service.py`)
   - 封装阿里云百炼TTS API
   - 处理音频文件生成和播放
   - 管理音频文件存储

2. **RAGSystem** (`core/rag_system.py`)
   - 集成TTS服务
   - 在生成回答后自动调用TTS

3. **配置文件** (`config.py`)
   - TTS参数配置
   - API Key管理

## 🎯 使用场景

1. **Web界面**：用户提问 → 大模型回答 → 自动语音播放
2. **命令行模式**：输入问题 → 生成回答 → 语音朗读
3. **API调用**：集成到其他应用 → 提供语音反馈

## 📞 支持

如有问题，请查看：
1. [阿里云百炼TTS文档](https://help.aliyun.com/zh/model-studio/developer-reference/tts-api)
2. [项目Issue](https://github.com/your-repo/issues)

---

Made with 💝 by wink-wink-wink555
