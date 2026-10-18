import sys
import contextlib
import io
import json
import shutil
import time
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
TEMP_ROOT = ROOT / ".test-tmp"
TEMP_ROOT.mkdir(exist_ok=True)

from extract_video import (  # noqa: E402
    Segment,
    join_text_fragments,
    VideoMetadata,
    load_env,
    main,
    parse_subtitle_text,
    safe_filename,
    split_text,
    summarize_locally,
    write_transcript_outputs,
)


class ExtractorTests(unittest.TestCase):
    def test_parse_srt(self):
        content = """1
00:00:00,000 --> 00:00:02,000
人才是第一资源。

2
00:00:02,500 --> 00:00:05,000
高校要提高人才自主培养质量。
"""
        segments = parse_subtitle_text(content)
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[1].start, 2.5)

    def test_parse_vtt(self):
        content = """WEBVTT

00:01.000 --> 00:03.000
建设教育强国。
"""
        segments = parse_subtitle_text(content)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "建设教育强国。")

    def test_parse_subtitle_skips_invalid_and_duplicate_cues(self):
        content = """1
00:00:00,000 --> 00:00:01,000
重复内容。

2
00:00:01,000 --> 00:00:02,000
重复 内容。

3
00:00:05,000 --> 00:00:04,000
非法时间轴。

4
00:00:05,000 --> 00:00:06,000
有效内容。
"""
        segments = parse_subtitle_text(content)
        self.assertEqual([segment.text for segment in segments], ["重复内容。", "有效内容。"])

    def test_local_summary(self):
        segments = [
            Segment(0, 5, "人才是第一资源。高校要提高人才自主培养质量。"),
            Segment(5, 10, "加快建设国家战略人才力量，培养高技能人才。"),
            Segment(10, 15, "以高质量人才工作服务中国式现代化建设。"),
        ]
        summary = summarize_locally(
            VideoMetadata(title="人才主题", source="test"),
            segments,
        )
        self.assertIn("一句话概括", summary)
        self.assertIn("分段时间线", summary)
        self.assertIn("人才", summary)

    def test_environment_parser(self):
        directory = TEMP_ROOT / f"env-{time.time_ns()}"
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = directory / ".env"
        path.write_text('VIDEO_URL="https://example.com/video"\nBOOL=true\n', encoding="utf-8")
        values = load_env(path)
        self.assertEqual(values["VIDEO_URL"], "https://example.com/video")
        self.assertEqual(values["BOOL"], "true")

    def test_safe_filename(self):
        self.assertEqual(safe_filename("a/b:c?d*"), "a_b_c_d_")
        self.assertTrue(safe_filename("CON").startswith("_"))
        self.assertEqual(safe_filename("CON.txt"), "_CON.txt")
        self.assertEqual(safe_filename("CON .txt"), "_CON .txt")

    def test_text_joining_and_long_line_chunking(self):
        self.assertEqual(join_text_fragments(["Hello,", "world."]), "Hello, world.")
        self.assertEqual(join_text_fragments(["你好，", "世界。"]), "你好，世界。")
        chunks = split_text("x" * 25_000, limit=12_000)
        self.assertEqual([len(chunk) for chunk in chunks], [12_000, 12_000, 1_000])

    def test_write_outputs(self):
        segments = [Segment(0, 1.5, "测试内容。")]
        output = TEMP_ROOT / f"output-{time.time_ns()}"
        self.addCleanup(shutil.rmtree, output, ignore_errors=True)
        write_transcript_outputs(
            output,
            VideoMetadata(title="测试", source="test"),
            segments,
            {"source": "unit-test"},
        )
        self.assertTrue((output / "transcript.srt").exists())
        self.assertTrue((output / "transcript.vtt").exists())
        self.assertTrue((output / "transcript.txt").exists())
        self.assertTrue((output / "transcript.json").exists())

    def test_writer_preserves_english_spacing(self):
        output = TEMP_ROOT / f"english-{time.time_ns()}"
        self.addCleanup(shutil.rmtree, output, ignore_errors=True)
        write_transcript_outputs(
            output,
            VideoMetadata(title="English", source="test"),
            [Segment(0, 1, "Hello,"), Segment(1, 2, "world.")],
        )
        self.assertEqual(
            (output / "transcript.txt").read_text(encoding="utf-8"),
            "Hello, world.",
        )

    def test_writer_skips_invalid_segments(self):
        output = TEMP_ROOT / f"invalid-segment-{time.time_ns()}"
        self.addCleanup(shutil.rmtree, output, ignore_errors=True)
        with contextlib.redirect_stdout(io.StringIO()):
            write_transcript_outputs(
                output,
                VideoMetadata(title="Invalid", source="test"),
                [Segment(2, 1, "bad"), Segment(1, 2, "good")],
            )
        payload = json.loads((output / "transcript.json").read_text(encoding="utf-8"))
        self.assertEqual([segment["text"] for segment in payload["segments"]], ["good"])

    def test_main_local_subtitle(self):
        directory = TEMP_ROOT / f"main-{time.time_ns()}"
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        subtitle = directory / "sample.srt"
        subtitle.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n本地测试内容。\n",
            encoding="utf-8",
        )
        output = directory / "output"
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(
                [
                    "--env-file",
                    str(directory / "missing.env"),
                    "--local-file",
                    str(subtitle),
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, 0)
        self.assertTrue((output / "sample" / "summary.md").is_file())

    def test_missing_explicit_local_file_does_not_use_url(self):
        directory = TEMP_ROOT / f"missing-{time.time_ns()}"
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        with self.assertRaisesRegex(SystemExit, "local-file"):
            main(
                [
                    "--env-file",
                    str(directory / "missing.env"),
                    "--local-file",
                    str(directory / "missing.mp4"),
                    "--url",
                    "https://example.com/video",
                ]
            )

    def test_bootstrap_boolean_parser(self):
        self.assertTrue(bootstrap.parse_bool("true", False))
        self.assertFalse(bootstrap.parse_bool("off", True))
        with self.assertRaises(ValueError):
            bootstrap.parse_bool("sometimes", False)

    def test_bootstrap_requirement_versions(self):
        directory = TEMP_ROOT / f"requirements-{time.time_ns()}"
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = directory / "requirements.txt"
        path.write_text("demo-package>=2.0\n", encoding="utf-8")
        with mock.patch.object(bootstrap.metadata, "version", return_value="2.0.0"):
            self.assertEqual(bootstrap.requirement_problems(path), [])
        with mock.patch.object(bootstrap.metadata, "version", return_value="1.9.9"):
            self.assertTrue(bootstrap.requirement_problems(path))


if __name__ == "__main__":
    unittest.main()
