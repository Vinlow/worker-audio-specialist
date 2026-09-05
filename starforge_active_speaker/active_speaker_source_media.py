"""Single raw decode for canonical pixels and the complete decoded video clock.

Additive throughput experiment. The frozen MediaProcessor still validates the
header, every packet, every decoded timestamp, audio samples and A/V origins.
Only its video frame-probe transport is supplied from the normalization pass.
No generated timestamps, frame skipping, hardware resizing or silent fallback.
"""
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess

from active_speaker_contracts import ContractViolation, content_identity
from active_speaker_media import MediaProcessor


class DecodedClock:
    TIME_BASE = re.compile(r"config in time_base: (\d+/\d+)")
    FRAME = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:")

    def __init__(self, stream_index, time_base):
        self.stream_index = stream_index
        self.time_base = Fraction(time_base)
        self.configured = False
        self.frames = []

    def consume(self, line):
        if "Parsed_showinfo_" not in line:
            return
        config = self.TIME_BASE.search(line)
        if config:
            if self.configured or Fraction(config[1]) != self.time_base:
                raise ContractViolation("decoded filter time base changed or differs from source")
            self.configured = True
        match = self.FRAME.search(line)
        if " n:" in line and not match:
            raise ContractViolation("decoded frame has no exact integer PTS")
        if match:
            if not self.configured or int(match[1]) != len(self.frames):
                raise ContractViolation("decoded clock frame sequence has a gap or reset")
            if len(self.frames) >= 864_000:
                raise ContractViolation("decoded raw frame count exceeds bounded workload")
            pts = int(match[2])
            # showinfo's pts_time is rounded to six significant digits. Retain
            # its exact integer PTS and independently checked filter time base;
            # serialize those exactly for the unchanged rational clock validator.
            self.frames.append({"stream_index": self.stream_index, "media_type": "video",
                "best_effort_timestamp": pts,
                "best_effort_timestamp_time": str(Fraction(pts) * self.time_base)})


class SingleDecodeMediaProcessor(MediaProcessor):
    def __init__(self, *, decoder="cpu", select_before_download=False, **kwargs):
        super().__init__(**kwargs)
        if decoder not in {"cpu", "nvdec"}:
            raise ContractViolation("unknown raw video decoder")
        self.decoder = decoder
        self.select_before_download = select_before_download
        self.prepared_source = None
        self.clock = None
        self.evidence = None

    def _run(self, command, label):
        # Reuse both frozen validators, including the first-frame origin check.
        # Cache scope is exact source path + explicit stream + show_frames only.
        if (self.clock and self.prepared_source == Path(command[-1]).resolve()
                and command[0] == self.ffprobe and "-show_frames" in command
                and "-select_streams" in command
                and command[command.index("-select_streams") + 1] == str(self.clock.stream_index)):
            return subprocess.CompletedProcess(command, 0,
                json.dumps({"frames": self.clock.frames}), "")
        return super()._run(command, label)

    def prepare(self, source: Path, directory: Path, stream_index: int):
        self.prepared_source, self.clock, self.evidence = None, None, None
        header = super()._run([self.ffprobe, "-v", "error", "-select_streams", str(stream_index),
            "-show_entries", "stream=index,codec_type,codec_name,pix_fmt,time_base", "-of", "json",
            str(source)], "single-decode stream selection")
        streams = json.loads(header.stdout).get("streams", [])
        if len(streams) != 1 or streams[0].get("codec_type") != "video":
            raise ContractViolation("single-decode selection is not one video stream")
        stream = streams[0]
        clock = DecodedClock(stream_index, stream["time_base"])
        decode_options, download_filter = self.decoder_options(stream, stream_index)
        command = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", "-nostats",
            "-xerror", "-n", "-copyts", "-threads", "4", *decode_options, "-i", str(source),
            "-map", f"0:{stream_index}", "-an", "-vf", self.filters(download_filter),
            "-filter_threads", "2", "-vsync", "cfr",
            "-c:v", "ffv1", "-level", "3", "-threads", "4", "-g", "1", "-f", "segment",
            "-segment_time", "120", "-reset_timestamps", "1", str(directory / "chunk-%04d.mkv")]
        self.decode(command, clock)
        chunks = sorted(directory.glob("chunk-*.mkv"))
        if not clock.frames or not chunks or len(chunks) > 31:
            raise ContractViolation("single raw decode returned incomplete evidence")
        self.prepared_source, self.clock = source.resolve(), clock
        self.evidence = {"method": "single-decode-showinfo-integer-pts-v1", "decoder": self.decoder,
            "codec": stream["codec_name"], "rawFrames": len(clock.frames),
            "rawClockIdentity": content_identity(clock.frames), "resize": "UNCHANGED_CPU_SCALE",
            "selectCanonicalFramesBeforeDownload": self.select_before_download}
        return chunks

    def filters(self, download_filter):
        clock = "showinfo=checksum=0,setpts=PTS-STARTPTS,"
        scale = "scale=w='min(640,iw)':h=-2,"
        fps = "fps=25:round=near,"
        # The raw clock still sees ALL frames. Selection uses unchanged PTS and
        # round=near; only unused pixel transfer/resize work moves after it.
        return (clock + fps + download_filter + scale if self.select_before_download else
            download_filter + clock + scale + fps) + "setpts=N/(25*TB)"

    def decoder_options(self, stream, stream_index):
        if self.decoder == "cpu":
            return [], ""
        codec = stream.get("codec_name")
        if codec not in {"h264", "hevc", "av1", "vp9"} or stream.get("pix_fmt") != "yuv420p":
            raise ContractViolation("NVDEC experiment supports explicit 8-bit 4:2:0 codecs only")
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            f"-c:{stream_index}", f"{codec}_cuvid"], "hwdownload,format=nv12,format=yuv420p,"

    @staticmethod
    def decode(command, clock):
        tail = []
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True) as process:
            try:
                for line in process.stderr:
                    clock.consume(line)
                    tail.append(line)
                    if len(tail) > 16:
                        tail.pop(0)
                if process.wait() != 0:
                    raise ContractViolation("single raw decode failed: " + "".join(tail)[-4096:])
            except BaseException:
                process.kill()
                process.wait()
                raise
