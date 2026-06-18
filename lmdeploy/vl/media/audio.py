# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pybase64

from .base import MediaIO


class AudioMediaIO(MediaIO[npt.NDArray]):
    """Audio loader for ASR-style multimodal models."""

    def __init__(self, sampling_rate: int = 16000, mono: bool = True, **kwargs):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.mono = mono
        self.kwargs = kwargs

    @staticmethod
    def _to_float32(audio: npt.NDArray, mono: bool = True) -> npt.NDArray:
        audio = np.asarray(audio)
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            audio = audio.astype(np.float32) / max(abs(info.min), info.max, 1)
        else:
            audio = audio.astype(np.float32)
        if mono and audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return audio

    @staticmethod
    def _resample(audio: npt.NDArray, src_sr: int, dst_sr: int) -> npt.NDArray:
        if src_sr == dst_sr:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)
        except ImportError:
            pass
        try:
            from scipy.signal import resample_poly
            gcd = np.gcd(src_sr, dst_sr)
            return resample_poly(audio, dst_sr // gcd, src_sr // gcd).astype(np.float32)
        except ImportError:
            raise ImportError('Please install librosa or scipy to resample audio inputs.')

    def _load_with_soundfile(self, file_obj) -> npt.NDArray:
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError('Please install soundfile via `pip install soundfile` to load audio files.')

        audio, src_sr = sf.read(file_obj)
        audio = self._to_float32(audio, mono=self.mono)
        return self._resample(audio, src_sr, self.sampling_rate)

    def load_bytes(self, data: bytes) -> npt.NDArray:
        return self._load_with_soundfile(BytesIO(data))

    def load_base64(self, media_type: str, data: str) -> npt.NDArray:
        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, filepath: Path) -> npt.NDArray:
        return self._load_with_soundfile(filepath)
