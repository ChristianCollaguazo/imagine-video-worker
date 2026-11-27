"""Bounded-memory SDR H.264 export from ComfyUI IMAGE tensors."""
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import av
from av.codec.codec import UnknownCodecError
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc, VideoReformatter
import numpy as np
import torch


NVENC_OPTIONS = {"preset": "p5", "rc": "vbr", "cq": "18"}
X264_OPTIONS = {"crf": "18"}
STAGES = ("frame_preparation", "device_to_cpu", "frame_wrapping", "pixel_conversion", "encoding", "muxing", "finalization")
MAX_PREP_PIXELS = 4_194_304


class ExportProfile:
    def __init__(self, images, fps, count):
        self.started = perf_counter()
        self.images = images
        self.fps = fps
        self.count = count
        self.seconds = dict.fromkeys(STAGES, 0.0)

    def measure(self, stage, operation, *args, **kwargs):
        started = perf_counter()
        try:
            return operation(*args, **kwargs)
        finally:
            self.seconds[stage] = self.seconds.get(stage, 0.0) + (perf_counter() - started)

    def report(self, encoder, fallback=False):
        elapsed = perf_counter() - self.started
        return {
            "encoder": encoder, "hardware_fallback": fallback, "seconds": round(elapsed, 6),
            "stages_seconds": {key: round(value, 6) for key, value in self.seconds.items()},
            "fps": self.fps, "frame_count": self.count, "duration_seconds": self.count / self.fps,
            "throughput_fps": round(self.count / elapsed, 3),
            "tensor_device": str(self.images.device), "tensor_dtype": str(self.images.dtype),
            "tensor_shape": list(self.images.shape), "tensor_strides": list(self.images.stride()),
        }


def _direct(stage, operation, *args, **kwargs):
    return operation(*args, **kwargs)


class FrameBatches:
    def __init__(self, images, count, profile=None):
        self.images = images.detach()
        self.profile = profile
        _, height, width, _ = images.shape
        self.size = min(count, max(1, MAX_PREP_PIXELS // (width * height)))
        self.shape = (self.size, height, width, 3)
        self.cuda = images.device.type == "cuda"
        self.numpy = images.device.type == "cpu" and images.dtype in (torch.float16, torch.float32, torch.float64)
        if self.numpy:
            self.source = self.images[..., :3].numpy()
            self.scratch = np.empty_like(self.source[:self.size], order="K")
            self.rgb = np.empty(self.shape, dtype=np.uint8)
        elif self.cuda:
            self.host = torch.empty(self.shape, dtype=torch.uint8, device="cpu", pin_memory=True)
            self.stream = torch.cuda.current_stream(images.device)
            if profile:
                self.events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

    def read(self, start, end):
        measure = self.profile.measure if self.profile else _direct
        if self.numpy:
            return measure("frame_preparation", self._quantize_numpy, start, end)
        batch = self.images[start:end, ..., :3]
        if not self.cuda:
            rgb = measure("frame_preparation", self._quantize_torch, batch)
            if rgb.device.type != "cpu":
                rgb = measure("device_to_cpu", rgb.cpu)
            return rgb.numpy()

        if self.profile:
            self.events[0].record(self.stream)
        rgb = self._quantize_torch(batch)
        if self.profile:
            self.events[1].record(self.stream)
        host = self.host[:end - start]
        host.copy_(rgb, non_blocking=True)
        if self.profile:
            self.events[2].record(self.stream)
        # Wait once per chunk, immediately before the CPU reads the pinned buffer.
        self.stream.synchronize()
        if self.profile:
            self.profile.seconds["frame_preparation"] += self.events[0].elapsed_time(self.events[1]) / 1000
            self.profile.seconds["device_to_cpu"] += self.events[1].elapsed_time(self.events[2]) / 1000
        return host.numpy()

    @staticmethod
    def _quantize_torch(batch):
        return (batch * 255).clamp_(0, 255).to(dtype=torch.uint8, memory_format=torch.contiguous_format)

    def _quantize_numpy(self, start, end):
        scratch = self.scratch[:end - start]
        rgb = self.rgb[:end - start]
        np.multiply(self.source[start:end], 255, out=scratch)
        np.clip(scratch, 0, 255, out=scratch)
        # Like torch.byte(), truncate the quantized values after clipping.
        with np.errstate(invalid="ignore"):
            if scratch.strides[-1] == scratch.itemsize:
                np.copyto(rgb, scratch, casting="unsafe")
            else:
                # H3's channel-first storage is cheaper to read one plane at a time.
                for channel in range(3):
                    np.copyto(rgb[..., channel], scratch[..., channel], casting="unsafe")
        return rgb


class HardwareEncoderError(RuntimeError):
    pass


def set_sdr_color(target):
    target.color_primaries = ColorPrimaries.BT709
    target.color_trc = ColorTrc.IEC61966_2_1
    target.colorspace = 1  # BT.709 matrix
    target.color_range = ColorRange.MPEG


def add_video_stream(output, encoder, fps, width, height):
    stream = output.add_stream(encoder, rate=fps)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    stream.options = NVENC_OPTIONS if encoder == "h264_nvenc" else X264_OPTIONS
    if encoder == "h264_nvenc":
        stream.codec_context.bit_rate = 0
    set_sdr_color(stream.codec_context)
    return stream


def _encode_file(path, images, fps, count, audio, encoder, check_cancel, profile):
    measure = profile.measure if profile else _direct
    output = av.open(str(path), mode="w", format="mp4", options={"movflags": "+faststart"})
    try:
        audio_stream = None
        if audio is not None:
            sample_rate = int(audio["sample_rate"])
            samples = audio["waveform"].detach()[0, :, :count * sample_rate // fps]
            layout = {1: "mono", 2: "stereo", 6: "5.1"}[samples.shape[0]]

        try:
            video_stream = measure("encoding", add_video_stream, output, encoder, fps, images.shape[2], images.shape[1])
            # Open the actual encoder before preparing frames or publishing the file.
            measure("encoding", video_stream.codec_context.open)
        except (UnknownCodecError, av.FFmpegError) as exc:
            if encoder == "h264_nvenc":
                raise HardwareEncoderError from exc
            raise
        if audio is not None:
            audio_stream = output.add_stream("aac", rate=sample_rate, layout=layout)

        batches = FrameBatches(images, count, profile)
        reformatter = VideoReformatter()

        def write_video(frame):
            try:
                packets = measure("encoding", video_stream.encode, frame)
            except av.FFmpegError as exc:
                if encoder == "h264_nvenc":
                    raise HardwareEncoderError from exc
                raise
            measure("muxing", output.mux, packets)

        def write_audio(block, pts):
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            frame.time_base = Fraction(1, sample_rate)
            frame.pts = pts
            measure("muxing", output.mux, measure("encoding", audio_stream.encode, frame))

        audio_offset = 0
        time_base = Fraction(1, fps)
        for start in range(0, count, batches.size):
            check_cancel()
            end = min(count, start + batches.size)
            for index, img in enumerate(batches.read(start, end), start):
                frame = measure("frame_wrapping", av.VideoFrame.from_numpy_buffer, img, format="rgb24")
                frame = measure("pixel_conversion", reformatter.reformat, frame, format="yuv420p", dst_colorspace=1)
                set_sdr_color(frame)
                frame.pts = index
                frame.time_base = time_base
                write_video(frame)
            if audio_stream:
                limit = min(end * sample_rate // fps, samples.shape[1])
                if limit > audio_offset:
                    segment = samples[:, audio_offset:limit]
                    if segment.device.type != "cpu":
                        segment = measure("device_to_cpu", segment.cpu)
                    block = measure("frame_preparation", lambda: segment.float().contiguous().numpy())
                    write_audio(block, audio_offset)
                    audio_offset = limit

        check_cancel()
        write_video(None)
        if audio_stream:
            measure("muxing", output.mux, measure("encoding", audio_stream.encode, None))
    finally:
        measure("finalization", output.close)


def export_video(path: Path, images, fps, count, *, audio=None, encoder="auto", check_cancel=lambda: None, profile=False):
    if (images.ndim != 4 or images.shape[-1] not in (3, 4) or min(images.shape[1:3]) < 2
            or any(dimension % 2 for dimension in images.shape[1:3]) or not images.is_floating_point()
            or fps < 1 or not 1 <= count <= len(images)):
        raise ValueError("Invalid H3 export dimensions, frame count, FPS, or dtype")
    if encoder not in ("auto", "h264_nvenc", "libx264"):
        raise ValueError("Unsupported H3 encoder")
    timings = ExportProfile(images, fps, count) if profile else None
    selected = "h264_nvenc" if encoder == "auto" else encoder
    with NamedTemporaryFile(prefix=".export-", suffix=".mp4", dir=path.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        try:
            _encode_file(staged, images, fps, count, audio, selected, check_cancel, timings)
        except HardwareEncoderError:
            if encoder != "auto":
                raise
            check_cancel()
            selected = "libx264"
            _encode_file(staged, images, fps, count, audio, selected, check_cancel, timings)
        check_cancel()
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    fallback = encoder == "auto" and selected == "libx264"
    if timings:
        return timings.report(selected, fallback)
    return {"encoder": selected, "hardware_fallback": fallback}
