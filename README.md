# 视频内容提取器

这个程序会读取 `.env` 中的 `VIDEO_URL`，自动检查运行环境、安装 Python 依赖、下载视频音频或站点字幕，然后生成转写稿、SRT 字幕和结构化摘要。

## 支持范围

- Windows 10/11、macOS、常见 x86_64/ARM64 Linux
- Python 3.10 或更高版本；启动器会尝试自动安装缺失的 Python
- 推荐 Python 3.12；启动器会优先选择兼容性更稳定的版本，并尝试安装缺失的 Python
- 优先使用站点公开字幕；没有字幕时使用 faster-whisper 做语音识别
- NVIDIA 显卡可用时自动尝试 GPU；CUDA 不可用时自动回退 CPU
- yt-dlp 支持的公开视频页面，以及本地音视频或 SRT/VTT 文件
- 不绕过 DRM、登录权限或付费权限

## 快速开始

1. 用文本编辑器打开 `.env`，修改 `VIDEO_URL`。
2. 在 Windows 上运行：

```powershell
cd "你的项目目录"
.\run.ps1
```

如果 PowerShell 阻止脚本执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

也可以双击 `run.cmd`。

macOS 或 Linux：

```bash
cd video-content-extractor
chmod +x run.sh
./run.sh
```

首次启动会创建 `.venv`，安装 `yt-dlp`、`faster-whisper` 和随包 FFmpeg，并在需要时下载 Whisper 模型。第二次启动会复用这些文件。

## 环境检查

只检查环境，不安装依赖、不开始提取：

```powershell
.\run.ps1 --check-only
```

检查结果同时写入 `environment-report.json`。安装日志位于 `logs/bootstrap.log`。

## 常用配置

编辑 `.env`：

```dotenv
VIDEO_URL=https://www.bilibili.com/video/BV1f8WszGEVY/
OUTPUT_DIR=output
WHISPER_MODEL=auto
WHISPER_LANGUAGE=zh
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
USE_AVAILABLE_SUBTITLES=true
COOKIES_FROM_BROWSER=
KEEP_AUDIO=false
AUTO_INSTALL_CUDA=true
```

- `WHISPER_MODEL=auto`：有 GPU 时选择 `large-v3`，只有 CPU 时选择 `small`
- `WHISPER_DEVICE=cpu`：强制使用 CPU
- `WHISPER_DEVICE=cuda`：强制尝试 GPU，失败时主程序仍会回退 CPU
- `COOKIES_FROM_BROWSER=edge`：B 站要求登录或报 412 时，从 Edge 读取登录状态
- `COOKIES_FILE=cookies.txt`：使用导出的 Netscape 格式 Cookie 文件
- `KEEP_AUDIO=true`：转写后保留 `work/` 中下载的音频
- `AUTO_INSTALL_CUDA=false`：检测到 NVIDIA 显卡时也不自动安装 CUDA 12 运行库

也可以临时覆盖参数：

```powershell
.\run.ps1 --url "https://www.bilibili.com/video/BV1f8WszGEVY/"
.\run.ps1 --device cpu
.\run.ps1 --no-subtitles
.\run.ps1 --local-file "D:\video.mp4"
.\run.ps1 --model medium --language zh
.\run.ps1 --no-cuda
```

## 输出文件

默认输出到 `output/<视频标题>/`：

如果两个不同视频标题相同，第二个结果会使用带视频 ID 和来源哈希的目录名，避免覆盖已有输出。

- `summary.md`：一句话概括、主要内容、分段时间线、关键词
- `transcript.txt`：便于阅读的纯文本转写
- `transcript.srt`：标准 SRT 字幕
- `transcript.vtt`：WebVTT 字幕
- `transcript.json`：包含元数据、时间戳和转写片段的完整结构化结果
- `.extractor.json`：记录输出来源与完成状态，用于安全复用或隔离同名目录

不配置 LLM 时使用本地抽取式摘要，完全不依赖额外 API。若需要语义质量更高的摘要，可以配置任意 OpenAI Chat Completions 兼容接口：

```dotenv
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=你的模型名
```

未填写 `LLM_MODEL` 时不会调用外部接口。API 调用失败时自动回退本地摘要。

## 自动化与可靠性设计

- 自动创建隔离的 `.venv`，避免污染系统 Python
- 已安装依赖满足版本要求时跳过重复安装，离线环境也能直接复用
- `.venv` 损坏或 Python 版本过旧时先重命名为 `.venv.invalid-*`，再尝试重建
- 启动前检查操作系统、Python、网络、磁盘、FFmpeg 和 NVIDIA 显卡
- 安装失败会保留详细日志并给出可操作错误
- GPU 转写在任一阶段失败时自动重建 CPU 模型并重新执行
- 优先使用已有字幕，减少下载量与转写时间
- 媒体缓存按视频来源隔离；零字节文件会被清理，完整缓存可安全复用
- 下载支持断点续传和重试
- 转写过程每 15 秒输出进度，并写入 `transcript.partial.jsonl` 作为中间记录
- 默认只下载音频，不保存完整视频，降低磁盘占用
- 下载的临时音频在转写成功或失败后都会清理，除非设置 `KEEP_AUDIO=true`
- 字幕时间轴会过滤倒序和零长度片段，输出文件使用临时文件原子替换
- SRT/VTT 可处理缺少空行、乱序时间轴和非法时间戳；VTT 标签会被清理
- 文件名同时限制字符数和 UTF-8 字节数，兼容 Linux、macOS 与 Windows 路径限制
- 不同来源的同名视频会隔离输出目录，不会互相覆盖
- LLM 请求限制响应大小，并拒绝会把 API 密钥带到其他站点的跨域重定向
- LLM 长文本按固定上限切块，超长单行也不会绕过限制
- 摘要失败不会丢失已生成的转写稿

## 限制

- “所有环境”在现实中不存在：程序覆盖主流 Windows、macOS 和 Linux，但仍需要可联网环境、可用的 Python 以及 yt-dlp 支持的站点。
- 直播、付费课程、DRM 视频、需要验证码或强登录校验的内容可能无法下载。
- 纯 CPU 转写 40 分钟视频可能需要较长时间；优先配置可用的 NVIDIA GPU。
- 说话人分离、画面文字 OCR 当前没有默认启用；主要输出是语音内容与结构化摘要。

## 自检

```powershell
python extract_video.py --self-test
python -m unittest discover -s tests -v
```

如果源码目录只读，可把测试临时目录指向可写位置：

```powershell
$env:VCE_TEST_TMP = "$env:TEMP\video-extractor-tests"
python -m unittest discover -s tests -v
```
