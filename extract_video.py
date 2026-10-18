#!/usr/bin/env python3
"""Download or transcribe a video and produce transcript and summary files."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from hashlib import sha256
from tempfile import mkstemp
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, OSError):
    pass
DEFAULT_OUTPUT = ROOT / "output"
WORK_ROOT = ROOT / "work"
MODEL_CACHE = ROOT / "models" / "whisper"
TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})"
)
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]")
WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
STOP_TERMS = {
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "这些",
    "那些",
    "就是",
    "可以",
    "需要",
    "进行",
    "一个",
    "以及",
    "通过",
    "不断",
    "必须",
    "应该",
    "已经",
    "还是",
    "但是",
    "因为",
    "所以",
    "如果",
    "对于",
    "关于",
    "同时",
    "其中",
    "在这",
    "在这",
    "有着",
    "能够",
    "具有",
    "方面",
    "问题",
    "时候",
    "大家",
    "今天",
    "首先",
    "其次",
    "最后",
    "谢谢",
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "for",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class VideoMetadata:
    title: str
    source: str
    video_id: str = ""
    duration: float | None = None
    uploader: str = ""
    description: str = ""
    subtitles: dict[str, object] | None = None
    automatic_captions: dict[str, object] | None = None


@dataclass
class Settings:
    url: str
    output_dir: Path
    model: str
    language: str
    device: str
    compute_type: str
    use_subtitles: bool
    cookies_from_browser: str
    cookies_file: str
    keep_audio: bool
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_timeout: int


def log(message: str) -> None:
    print(f"[extractor] {message}", flush=True)


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = value.strip().casefold()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def env_value(env: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key, env.get(key, default))


def resolve_path(value: str, default: Path) -> Path:
    if not value:
        return default
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else ROOT / path


def pre_read_env_file(argv: list[str]) -> Path:
    for index in range(len(argv) - 1, -1, -1):
        item = argv[index]
        if item == "--env-file" and index + 1 < len(argv):
            return resolve_path(argv[index + 1], ROOT / ".env")
        if item.startswith("--env-file="):
            return resolve_path(item.split("=", 1)[1], ROOT / ".env")
    return ROOT / ".env"


def parse_args(argv: list[str], env: dict[str, str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a video transcript and summary using yt-dlp and faster-whisper."
    )
    parser.add_argument("--url", default=env_value(env, "VIDEO_URL"))
    parser.add_argument("--local-file", default="", help="Use a local media or subtitle file")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--output", default=env_value(env, "OUTPUT_DIR", "output"))
    parser.add_argument("--model", default=env_value(env, "WHISPER_MODEL", "auto"))
    parser.add_argument("--language", default=env_value(env, "WHISPER_LANGUAGE", "zh"))
    parser.add_argument("--device", default=env_value(env, "WHISPER_DEVICE", "auto"))
    parser.add_argument(
        "--compute-type",
        default=env_value(env, "WHISPER_COMPUTE_TYPE", "auto"),
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Ignore site subtitles and always transcribe the audio",
    )
    parser.add_argument("--cookies-from-browser", default=env_value(env, "COOKIES_FROM_BROWSER"))
    parser.add_argument("--cookies-file", default=env_value(env, "COOKIES_FILE"))
    parser.add_argument(
        "--keep-audio",
        action="store_true",
    )
    parser.add_argument("--force-transcribe", action="store_true")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def safe_filename(value: str, fallback: str = "video") -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    value = value[:120].rstrip(" .")
    if not value:
        value = fallback
    stem = value.split(".", 1)[0].rstrip(" .").upper()
    if stem in WINDOWS_RESERVED:
        value = f"_{value}"
    return value


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def join_text_fragments(fragments: list[str]) -> str:
    result = ""
    for raw_fragment in fragments:
        fragment = raw_fragment.strip()
        if not fragment:
            continue
        if result:
            previous = result[-1]
            current = fragment[0]
            resumes_after_ascii_punctuation = previous in ",.;:!?)]}" and ASCII_WORD_RE.match(
                current
            )
            if (
                ASCII_WORD_RE.match(previous) and ASCII_WORD_RE.match(current)
            ) or resumes_after_ascii_punctuation:
                result += " "
        result += fragment
    return result


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"&nbsp;", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_timestamp(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def format_timestamp(seconds: float, vtt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def compact_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_subtitle_text(content: str) -> list[Segment]:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", content)
    segments: list[Segment] = []
    previous = ""
    for block in blocks:
        lines = [line.strip("\ufeff ") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((i for i, line in enumerate(lines) if TIME_RE.search(line)), -1)
        if time_index < 0:
            continue
        match = TIME_RE.search(lines[time_index])
        if not match:
            continue
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        text = join_text_fragments(lines[time_index + 1 :])
        signature = re.sub(r"\s+", "", text).casefold()
        if not text or signature == previous or text == "WEBVTT":
            continue
        previous = signature
        segments.append(
            Segment(
                start=start,
                end=end,
                text=text,
            )
        )
    return segments


def write_transcript_outputs(
    output_dir: Path,
    metadata: VideoMetadata,
    segments: list[Segment],
    transcription_info: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_segments: list[Segment] = []
    for segment in segments:
        start = float(segment.start)
        end = float(segment.end)
        text = clean_text(segment.text) if isinstance(segment.text, str) else ""
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            log(f"Skipping invalid transcript segment: {segment.start} -> {segment.end}")
            continue
        if text:
            normalized_segments.append(Segment(max(0.0, start), end, text))
    if not normalized_segments:
        raise ValueError("Cannot write an empty transcript")

    srt_parts: list[str] = []
    vtt_parts = ["WEBVTT", ""]
    for index, segment in enumerate(normalized_segments, 1):
        srt_parts.append(
            f"{index}\n"
            f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            f"{segment.text}\n"
        )
        vtt_parts.extend(
            [
                f"{format_timestamp(segment.start, vtt=True)} --> "
                f"{format_timestamp(segment.end, vtt=True)}",
                segment.text,
                "",
            ]
        )

    atomic_write_text(output_dir / "transcript.srt", "\n".join(srt_parts))
    atomic_write_text(output_dir / "transcript.vtt", "\n".join(vtt_parts))

    paragraphs: list[str] = []
    buffer: list[str] = []
    previous_end = 0.0
    for segment in normalized_segments:
        if buffer and segment.start - previous_end > 2:
            paragraphs.append(join_text_fragments(buffer))
            buffer = []
        buffer.append(segment.text)
        previous_end = segment.end
        if segment.text.rstrip().endswith(("。", "！", "？", "；", ".", "!", "?", ";")):
            paragraphs.append(join_text_fragments(buffer))
            buffer = []
    if buffer:
        paragraphs.append(join_text_fragments(buffer))

    atomic_write_text(
        output_dir / "transcript.txt",
        "\n\n".join(part for part in paragraphs if part),
    )

    payload = {
        "metadata": {
            "title": metadata.title,
            "source": metadata.source,
            "video_id": metadata.video_id,
            "duration": metadata.duration,
            "uploader": metadata.uploader,
            "description": metadata.description,
            "subtitle_languages": sorted((metadata.subtitles or {}).keys()),
            "automatic_caption_languages": sorted(
                (metadata.automatic_captions or {}).keys()
            ),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "transcription": transcription_info or {},
        "segments": [asdict(segment) for segment in normalized_segments],
    }
    atomic_write_text(
        output_dir / "transcript.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def make_units(segments: list[Segment]) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    buffer: list[Segment] = []
    char_count = 0

    def flush() -> None:
        nonlocal buffer, char_count
        if not buffer:
            return
        text = join_text_fragments([segment.text for segment in buffer])
        if text:
            units.append(
                {
                    "start": buffer[0].start,
                    "end": buffer[-1].end,
                    "text": text,
                }
            )
        buffer = []
        char_count = 0

    for segment in segments:
        buffer.append(segment)
        char_count += len(segment.text)
        ended = segment.text.rstrip().endswith(("。", "！", "？", "；", ".", "!", "?", ";"))
        if char_count >= 180 or (char_count >= 90 and ended):
            flush()
    flush()
    return units


def term_counter(text: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        for size in (2, 3, 4):
            for index in range(max(0, len(run) - size + 1)):
                term = run[index : index + size]
                if term not in STOP_TERMS:
                    counter[term] += 1
    for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text):
        lowered = term.lower()
        if lowered not in STOP_TERMS:
            counter[lowered] += 1
    return counter


def top_terms(text: str, limit: int = 20) -> list[str]:
    counts = term_counter(text)
    ranked = sorted(
        counts.items(),
        key=lambda item: (item[1] * math.log(2 + len(item[0])), len(item[0])),
        reverse=True,
    )
    selected: list[str] = []
    for term, _ in ranked:
        if any(term in existing or existing in term for existing in selected):
            continue
        selected.append(term)
        if len(selected) >= limit:
            break
    return selected


def summarize_locally(metadata: VideoMetadata, segments: list[Segment]) -> str:
    units = make_units(segments)
    if not units:
        raise RuntimeError("No speech segments were produced, so no summary can be generated")

    all_text = join_text_fragments([str(unit["text"]) for unit in units])
    terms = top_terms(all_text, 14)
    counts = term_counter(all_text)
    scored: list[tuple[float, dict[str, object]]] = []
    for index, unit in enumerate(units):
        text = str(unit["text"])
        unit_terms = term_counter(text)
        score = sum(
            math.log(1 + counts[term]) * count
            for term, count in unit_terms.items()
            if term in counts
        )
        score /= math.sqrt(max(1, len(text)))
        if index < max(3, len(units) // 12):
            score *= 1.12
        if len(text) < 18:
            score *= 0.5
        scored.append((score, unit))

    chosen: list[dict[str, object]] = []
    for _, unit in sorted(scored, key=lambda item: item[0], reverse=True):
        if len(chosen) >= 9:
            break
        if any(
            abs(float(unit["start"]) - float(existing["start"])) < 8
            for existing in chosen
        ):
            continue
        chosen.append(unit)
    chosen.sort(key=lambda item: float(item["start"]))

    bin_size = 300
    bins: dict[int, list[tuple[float, dict[str, object]]]] = {}
    for score, unit in scored:
        bin_index = int(float(unit["start"]) // bin_size)
        bins.setdefault(bin_index, []).append((score, unit))

    timeline_lines: list[str] = []
    for bin_index in sorted(bins):
        best = sorted(bins[bin_index], key=lambda item: item[0], reverse=True)[:2]
        best.sort(key=lambda item: float(item[1]["start"]))
        for _, unit in best:
            timeline_lines.append(
                f"- **{compact_timestamp(float(unit['start']))}-"
                f"{compact_timestamp(float(unit['end']))}** {unit['text']}"
            )

    keyword_text = "、".join(f"“{term}”" for term in terms[:7]) or "视频主题"
    overview = (
        f"本视频围绕{keyword_text}等主题展开，系统梳理了相关背景、重点任务、"
        "实践路径与结论。以下内容按视频时间顺序提炼，便于快速阅读和复核。"
    )
    main_lines = [
        f"{index}. **[{compact_timestamp(float(unit['start']))}]** {unit['text']}"
        for index, unit in enumerate(chosen, 1)
    ]
    return "\n".join(
        [
            f"# {metadata.title}",
            "",
            f"- 来源：{metadata.source}",
            f"- 转写片段数：{len(segments)}",
            "",
            "## 一句话概括",
            "",
            overview,
            "",
            "## 主要内容",
            "",
            *main_lines,
            "",
            "## 分段时间线",
            "",
            *(timeline_lines or ["- 未生成时间线。"]),
            "",
            "## 关键词",
            "",
            "、".join(terms[:20]),
            "",
            "## 说明",
            "",
            "本摘要由本地抽取式算法生成，优先保留高频主题和时间顺序；"
            "如需更强的语义概括，可在 `.env` 中配置兼容的 LLM API。",
            "",
        ]
    )


def transcript_for_llm(segments: list[Segment]) -> str:
    return "\n".join(
        f"[{compact_timestamp(segment.start)}-{compact_timestamp(segment.end)}] "
        f"{segment.text}"
        for segment in segments
    )


def split_text(text: str, limit: int = 12000) -> list[str]:
    if limit < 1:
        raise ValueError("Text chunk limit must be positive")
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current:
            chunks.append("\n".join(current))
            current = []
            current_size = 0

    for raw_line in text.splitlines():
        line = raw_line
        while len(line) > limit:
            flush()
            chunks.append(line[:limit])
            line = line[limit:]
        line_size = len(line) + (1 if current else 0)
        if current and current_size + line_size > limit:
            flush()
            line_size = len(line)
        current.append(line)
        current_size += line_size
    flush()
    return chunks


def llm_chat(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
) -> str:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    parsed_endpoint = urlparse(endpoint)
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
        raise ValueError(f"Invalid LLM_BASE_URL: {base_url}")

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是严谨的视频内容编辑。只依据提供的材料总结，"
                        "不得补充材料中没有的事实。使用简体中文和 Markdown。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1800,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "video-content-extractor/1.0",
        },
        method="POST",
    )
    context = ssl.create_default_context()
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                raw_response = response.read(8 * 1024 * 1024 + 1)
            if len(raw_response) > 8 * 1024 * 1024:
                raise ValueError("LLM response exceeded the 8 MiB safety limit")
            result = json.loads(raw_response.decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("LLM returned an empty response")
            return content.strip()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(2048).decode("utf-8", errors="replace").strip()
            except (OSError, AttributeError):
                detail = ""
            last_error = RuntimeError(
                f"HTTP {exc.code}: {detail or exc.reason}"
            )
            retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
            if not retryable:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            break
        if attempt < 3:
            time.sleep(2 * attempt)
    raise RuntimeError(f"LLM request failed: {last_error}")


def summarize_with_llm(
    settings: Settings,
    metadata: VideoMetadata,
    segments: list[Segment],
) -> str:
    text = transcript_for_llm(segments)
    chunks = split_text(text)
    if not chunks:
        raise RuntimeError("Transcript is empty; cannot call the LLM")
    chunk_summaries: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        log(f"Summarizing transcript chunk {index}/{len(chunks)} with LLM")
        prompt = (
            f"视频标题：{metadata.title}\n"
            f"这是第 {index}/{len(chunks)} 段带时间戳的转写。请提炼 4-8 条要点，"
            "保留关键人物、机构、政策概念、数据、案例和结论，并保留时间戳。"
            "不要写套话。\n\n"
            f"{chunk}"
        )
        chunk_summaries.append(
            llm_chat(
                settings.llm_api_key,
                settings.llm_base_url,
                settings.llm_model,
                prompt,
                settings.llm_timeout,
            )
        )

    combined = "\n\n".join(
        f"### 片段 {index}\n{summary}"
        for index, summary in enumerate(chunk_summaries, 1)
    )
    final_prompt = (
        f"视频标题：{metadata.title}\n"
        "请把下面的分段摘要合并成最终摘要，要求：\n"
        "1. 先给出一段不超过 220 字的总述；\n"
        "2. 再给出 6-10 条按时间顺序排列的主要内容，每条带时间戳；\n"
        "3. 单独列出关键词；\n"
        "4. 最后给出 2-4 句话的结论；\n"
        "5. 合并重复内容，不要遗漏不同讲解者的核心观点。\n\n"
        f"{combined}"
    )
    body = llm_chat(
        settings.llm_api_key,
        settings.llm_base_url,
        settings.llm_model,
        final_prompt,
        settings.llm_timeout,
    )
    return "\n".join(
        [
            f"# {metadata.title}",
            "",
            f"- 来源：{metadata.source}",
            f"- 转写片段数：{len(segments)}",
            "",
            body,
            "",
        ]
    )


def first_entry(info: dict[str, object]) -> dict[str, object]:
    if not isinstance(info, dict):
        raise RuntimeError("yt-dlp returned no video information")
    entries = info.get("entries")
    if entries:
        if isinstance(entries, dict):
            entries = entries.values()
        first = next((entry for entry in entries if isinstance(entry, dict)), None)
        if first is None:
            raise RuntimeError("The playlist or URL contained no downloadable video")
        return first
    return info


def optional_duration(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration >= 0 else None


def import_ytdlp():
    try:
        import yt_dlp  # type: ignore

        return yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is missing. Run run.ps1 or run.sh to install dependencies."
        ) from exc


def ytdlp_options(settings: Settings) -> dict[str, object]:
    options: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 45,
        "continuedl": True,
        "no_color": True,
    }
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        options["ffmpeg_location"] = ffmpeg
    if settings.cookies_file:
        cookie_path = resolve_path(settings.cookies_file, ROOT / settings.cookies_file)
        if not cookie_path.is_file():
            raise RuntimeError(f"COOKIES_FILE does not exist: {cookie_path}")
        options["cookiefile"] = str(cookie_path)
    if settings.cookies_from_browser:
        browser = settings.cookies_from_browser.strip()
        options["cookiesfrombrowser"] = (browser, None, None, None)
    return options


def friendly_download_error(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if "412" in lowered or "403" in lowered or "login" in lowered:
        return (
            f"{message}\n"
            "The site may require browser cookies. Set COOKIES_FROM_BROWSER=edge "
            "or COOKIES_FILE=cookies.txt in .env and retry."
        )
    if "unsupported url" in lowered:
        return f"{message}\nThe URL is not supported by the installed yt-dlp version."
    return message


def inspect_video(url: str, settings: Settings) -> VideoMetadata:
    yt_dlp = import_ytdlp()
    try:
        with yt_dlp.YoutubeDL(ytdlp_options(settings)) as downloader:
            info = downloader.extract_info(url, download=False)
        info = first_entry(info)
    except Exception as exc:
        raise RuntimeError(friendly_download_error(exc)) from exc
    return VideoMetadata(
        title=str(info.get("title") or "未命名视频"),
        source=url,
        video_id=str(info.get("id") or ""),
        duration=optional_duration(info.get("duration")),
        uploader=str(info.get("uploader") or ""),
        description=str(info.get("description") or ""),
        subtitles=info.get("subtitles") if isinstance(info.get("subtitles"), dict) else {},
        automatic_captions=(
            info.get("automatic_captions")
            if isinstance(info.get("automatic_captions"), dict)
            else {}
        ),
    )


def choose_subtitle(metadata: VideoMetadata) -> tuple[str, bool] | None:
    preferred = ["zh-Hans", "zh-CN", "zh", "zh-Hant", "zh-TW", "en", "en-US"]
    for automatic, collection in (
        (False, metadata.subtitles or {}),
        (True, metadata.automatic_captions or {}),
    ):
        if not collection:
            continue
        available = {str(language).casefold(): str(language) for language in collection}
        for language in preferred:
            actual_language = available.get(language.casefold())
            if actual_language:
                return actual_language, automatic
        chinese = next(
            (
                actual_language
                for normalized, actual_language in available.items()
                if normalized.startswith("zh")
            ),
            None,
        )
        if chinese:
            return chinese, automatic
    return None


def download_subtitle(
    url: str,
    settings: Settings,
    work_dir: Path,
    language: str,
    automatic: bool,
) -> list[Segment] | None:
    yt_dlp = import_ytdlp()
    work_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in work_dir.glob("source*"):
        if stale_path.is_file() and stale_path.suffix.lower() in {".srt", ".vtt"}:
            stale_path.unlink(missing_ok=True)
    options = ytdlp_options(settings)
    options.update(
        {
            "skip_download": True,
            "writesubtitles": not automatic,
            "writeautomaticsub": automatic,
            "subtitleslangs": [language],
            "subtitlesformat": "srt/vtt/best",
            "outtmpl": str(work_dir / "source.%(ext)s"),
        }
    )
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            result = downloader.download([url])
        if result not in (None, 0):
            raise RuntimeError(f"yt-dlp subtitle download exited with code {result}")
    except Exception as exc:
        log(f"Subtitle download failed; falling back to audio: {friendly_download_error(exc)}")
        return None

    candidates = sorted(
        [
            path
            for path in work_dir.glob("source*")
            if path.suffix.lower() in {".srt", ".vtt"}
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            log(f"Could not read subtitle file {path.name}: {exc}")
            continue
        segments = parse_subtitle_text(content)
        if segments:
            log(f"Using site subtitle: {path.name} ({len(segments)} cues)")
            return segments
    return None


def find_ffmpeg() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def download_media(url: str, settings: Settings, work_dir: Path) -> tuple[Path, dict[str, object]]:
    yt_dlp = import_ytdlp()
    work_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    options = ytdlp_options(settings)
    options.update(
        {
            "format": "bestaudio/best",
            "outtmpl": str(work_dir / "media.%(ext)s"),
            "overwrites": False,
        }
    )
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            info = first_entry(info)
            prepared = Path(downloader.prepare_filename(info))
    except Exception as exc:
        raise RuntimeError(friendly_download_error(exc)) from exc

    if prepared.is_file() and prepared.stat().st_size > 0:
        return prepared, info
    candidates = [
        path
        for path in work_dir.glob("media.*")
        if path.suffix.lower()
        not in {".part", ".ytdl", ".json", ".srt", ".vtt", ".description"}
        and path.is_file()
        and path.stat().st_size > 0
        and path.stat().st_mtime >= started_at - 2
    ]
    if not candidates:
        raise RuntimeError("Download completed but no current media file was found")
    return max(candidates, key=lambda path: path.stat().st_mtime), info


def detect_device(requested: str) -> tuple[str, str]:
    if requested == "cpu":
        return "cpu", choose_compute_type("cpu")
    try:
        import ctranslate2  # type: ignore

        count = ctranslate2.get_cuda_device_count()
        if count > 0:
            types = ctranslate2.get_supported_compute_types("cuda")
        else:
            types = set()
    except Exception:
        count = 0
        types = set()

    if requested == "cuda" or (requested == "auto" and count > 0):
        if "float16" in types:
            return "cuda", "float16"
        if "int8_float16" in types:
            return "cuda", "int8_float16"
        if "float32" in types:
            return "cuda", "float32"
        if requested == "cuda":
            raise RuntimeError(
                "CUDA was requested but no supported compute type is available. "
                "Use --device cpu or repair the CUDA runtime."
            )
    return "cpu", choose_compute_type("cpu")


def choose_compute_type(device: str) -> str:
    try:
        import ctranslate2  # type: ignore

        types = ctranslate2.get_supported_compute_types(device)
    except Exception:
        types = set()
    for candidate in (
        ("int8", "int8_float32", "float32")
        if device == "cpu"
        else ("float16", "int8_float16", "float32")
    ):
        if candidate in types:
            return candidate
    return "int8" if device == "cpu" else "float16"


def transcribe_audio(
    audio_path: Path,
    settings: Settings,
    work_dir: Path,
) -> tuple[list[Segment], dict[str, object]]:
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"Audio input is missing or empty: {audio_path}")
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is missing. Run run.ps1 or run.sh to install dependencies."
        ) from exc

    requested_device = settings.device.lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("WHISPER_DEVICE must be auto, cpu, or cuda")
    device, compute_type = detect_device(requested_device)
    if settings.compute_type.lower() != "auto":
        compute_type = settings.compute_type
    model_name = settings.model
    if model_name.lower() == "auto":
        model_name = "large-v3" if device == "cuda" else "small"

    attempts = [(device, compute_type)]
    if device == "cuda":
        attempts.append(("cpu", choose_compute_type("cpu")))

    last_error: Exception | None = None
    for attempt_index, (attempt_device, attempt_compute) in enumerate(attempts, 1):
        log(
            f"Loading Whisper model {model_name} for {attempt_device} "
            f"({attempt_compute}), attempt {attempt_index}/{len(attempts)}"
        )
        try:
            model = WhisperModel(
                model_name,
                device=attempt_device,
                compute_type=attempt_compute,
                download_root=str(MODEL_CACHE),
                cpu_threads=max(1, min(os.cpu_count() or 4, 8)),
            )
            segment_iterator, info = model.transcribe(
                str(audio_path),
                language=settings.language or None,
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=True,
            )
            segments: list[Segment] = []
            started = time.monotonic()
            last_report = started
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            partial_path = work_dir / "transcript.partial.jsonl"
            with partial_path.open("w", encoding="utf-8") as partial:
                for raw_segment in segment_iterator:
                    text = clean_text(raw_segment.text)
                    if not text:
                        continue
                    segment = Segment(float(raw_segment.start), float(raw_segment.end), text)
                    segments.append(segment)
                    partial.write(json.dumps(asdict(segment), ensure_ascii=False) + "\n")
                    partial.flush()
                    now = time.monotonic()
                    if now - last_report >= 15:
                        progress = (
                            f"{segment.end / duration * 100:.1f}%"
                            if duration
                            else f"{len(segments)} segments"
                        )
                        log(f"Transcription progress: {compact_timestamp(segment.end)} ({progress})")
                        last_report = now
            if not segments:
                raise RuntimeError("The audio contained no recognizable speech")
            partial_path.unlink(missing_ok=True)
            transcription_info = {
                "device": attempt_device,
                "compute_type": attempt_compute,
                "model": model_name,
                "language": getattr(info, "language", settings.language),
                "language_probability": getattr(info, "language_probability", None),
                "duration": duration,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
            return segments, transcription_info
        except Exception as exc:
            last_error = exc
            log(f"Transcription attempt failed: {exc}")
            if attempt_device == "cuda":
                log("CUDA is unavailable or incomplete; retrying with CPU")
            else:
                break
    raise RuntimeError(f"Transcription failed: {last_error}")


def is_subtitle_file(path: Path) -> bool:
    return path.suffix.lower() in {".srt", ".vtt"}


def self_test() -> int:
    sample = """1
00:00:00,000 --> 00:00:02,000
人才是第一资源。

2
00:00:02,000 --> 00:00:05,000
高校要全面提高人才自主培养质量。
"""
    segments = parse_subtitle_text(sample)
    assert len(segments) == 2
    assert abs(segments[1].start - 2.0) < 0.001
    metadata = VideoMetadata(title="自检", source="self-test")
    summary = summarize_locally(metadata, segments)
    assert "人才" in summary
    assert "分段时间线" in summary
    print("Self-test passed")
    return 0


def is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.netloc)


def resolve_local_input(explicit_value: str, url: str) -> Path | None:
    explicit = explicit_value.strip()
    if explicit:
        path = Path(os.path.expandvars(os.path.expanduser(explicit)))
        if not path.is_file():
            raise SystemExit(f"--local-file is not a readable file: {path}")
        return path.resolve()

    candidate = url.strip()
    if not candidate:
        raise SystemExit(
            "VIDEO_URL is empty. Put an http(s) URL or local file path in .env."
        )
    path = Path(os.path.expandvars(os.path.expanduser(candidate)))
    if path.is_file():
        return path.resolve()
    if is_http_url(candidate):
        return None
    if re.match(r"^[A-Za-z]:[\\/]", candidate) or "/" in candidate or "\\" in candidate:
        raise SystemExit(f"VIDEO_URL points to a local file that does not exist: {path}")
    raise SystemExit(
        "VIDEO_URL is invalid. Use an http(s) URL or an existing local file path."
    )


def work_directory_name(metadata: VideoMetadata) -> str:
    identity = metadata.video_id.strip()
    if not identity:
        source_hash = sha256(metadata.source.encode("utf-8", errors="replace")).hexdigest()[:12]
        identity = f"{metadata.title}-{source_hash}"
    return safe_filename(identity, fallback="media")


def build_settings(args: argparse.Namespace, env: dict[str, str]) -> Settings:
    model = args.model.strip() or "auto"
    device = args.device.strip().lower() or "auto"
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("WHISPER_DEVICE must be auto, cpu, or cuda")
    timeout_text = env_value(env, "LLM_TIMEOUT", "300").strip()
    try:
        llm_timeout = int(timeout_text)
    except ValueError as exc:
        raise ValueError("LLM_TIMEOUT must be an integer between 1 and 3600") from exc
    if not 1 <= llm_timeout <= 3600:
        raise ValueError("LLM_TIMEOUT must be between 1 and 3600 seconds")
    llm_base_url = env_value(
        env,
        "LLM_BASE_URL",
        env_value(env, "OPENAI_BASE_URL", "https://api.openai.com/v1"),
    ).strip()
    llm_api_key = (
        env_value(env, "LLM_API_KEY") or env_value(env, "OPENAI_API_KEY")
    ).strip()
    llm_model = env_value(
        env, "LLM_MODEL", env_value(env, "SUMMARY_MODEL")
    ).strip()
    if llm_api_key and not llm_model:
        log("LLM_API_KEY is set but LLM_MODEL is empty; using the local summary")
    if llm_model and not llm_api_key:
        log("LLM_MODEL is set but no API key is configured; using the local summary")

    return Settings(
        url=args.url.strip(),
        output_dir=resolve_path(args.output, DEFAULT_OUTPUT),
        model=model,
        language=args.language.strip(),
        device=device,
        compute_type=args.compute_type.strip().lower() or "auto",
        use_subtitles=not args.no_subtitles
        and parse_bool(env_value(env, "USE_AVAILABLE_SUBTITLES"), True),
        cookies_from_browser=args.cookies_from_browser.strip(),
        cookies_file=args.cookies_file.strip(),
        keep_audio=args.keep_audio
        or parse_bool(env_value(env, "KEEP_AUDIO"), False),
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url or "https://api.openai.com/v1",
        llm_model=llm_model,
        llm_timeout=llm_timeout,
    )


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    env_file = pre_read_env_file(raw_args)
    env = load_env(env_file)
    args = parse_args(raw_args, env)
    if args.self_test:
        return self_test()
    settings = build_settings(args, env)

    local_path = resolve_local_input(args.local_file, settings.url)
    downloaded_audio_path: Path | None = None

    if local_path and is_subtitle_file(local_path):
        metadata = VideoMetadata(title=local_path.stem, source=str(local_path))
        segments = parse_subtitle_text(
            local_path.read_text(encoding="utf-8-sig", errors="replace")
        )
        if not segments:
            raise SystemExit(f"No valid subtitle cues were found in: {local_path}")
        transcription_info = {"source": "local-subtitle", "model": None}
        audio_path: Path | None = None
    else:
        if local_path:
            metadata = VideoMetadata(title=local_path.stem, source=str(local_path))
        else:
            log("Reading video metadata")
            metadata = inspect_video(settings.url, settings)
        log(f"Video: {metadata.title}")
        if metadata.duration:
            log(f"Duration: {compact_timestamp(metadata.duration)}")

        work_name = work_directory_name(metadata)
        work_dir = WORK_ROOT / work_name
        work_dir.mkdir(parents=True, exist_ok=True)
        segments: list[Segment] | None = None
        if not local_path and settings.use_subtitles and not args.force_transcribe:
            choice = choose_subtitle(metadata)
            if choice:
                language, automatic = choice
                log(f"Site subtitle found: {language} (automatic={automatic})")
                segments = download_subtitle(
                    settings.url,
                    settings,
                    work_dir,
                    language,
                    automatic,
                )
        if segments:
            audio_path = None
            transcription_info = {"source": "site-subtitle", "model": None}
        else:
            if local_path:
                audio_path = local_path
            else:
                log("No usable site subtitle; downloading audio")
                audio_path, downloaded_info = download_media(
                    settings.url,
                    settings,
                    work_dir,
                )
                downloaded_audio_path = audio_path
                metadata.duration = metadata.duration or optional_duration(
                    downloaded_info.get("duration")
                )
                log(f"Downloaded audio: {audio_path}")
            try:
                segments, transcription_info = transcribe_audio(
                    audio_path, settings, work_dir
                )
            finally:
                if downloaded_audio_path and not settings.keep_audio:
                    try:
                        downloaded_audio_path.unlink(missing_ok=True)
                    except OSError as exc:
                        log(f"Could not remove downloaded audio: {exc}")

    output_dir = settings.output_dir / safe_filename(metadata.title)
    log(f"Writing transcript to: {output_dir}")
    write_transcript_outputs(output_dir, metadata, segments, transcription_info)

    if settings.llm_api_key and settings.llm_model:
        try:
            log(f"Generating LLM summary with model: {settings.llm_model}")
            summary = summarize_with_llm(settings, metadata, segments)
        except Exception as exc:
            log(f"LLM summary failed; using local summary: {exc}")
            summary = summarize_locally(metadata, segments)
    else:
        log("Generating local structured summary")
        summary = summarize_locally(metadata, segments)
    atomic_write_text(output_dir / "summary.md", summary)


    print("\nExtraction completed")
    print(f"Summary:   {output_dir / 'summary.md'}")
    print(f"Transcript:{output_dir / 'transcript.txt'}")
    print(f"SRT:       {output_dir / 'transcript.srt'}")
    print(f"JSON:      {output_dir / 'transcript.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)

