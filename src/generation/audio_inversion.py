from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F


SUPPORTED_AUDIO_METHODS = ("griffinlim", "istft", "gradient", "decorsiere")


@dataclass
class AudioGeometry:
    """!
    @brief STFT/Mel settings needed to invert a spectrogram into audio.
    """

    hop_length: int
    sample_rate: int = 22050
    n_fft: int = 512
    spectrogram_type: str = "linear"
    n_mels: int = 256

    def __post_init__(self) -> None:
        """!
        @brief Normalize numeric fields and validate the spectrogram type.
        """
        self.hop_length = int(self.hop_length)
        self.sample_rate = int(self.sample_rate)
        self.n_fft = int(self.n_fft)
        self.n_mels = int(self.n_mels)
        self.spectrogram_type = str(self.spectrogram_type).strip().lower()
        if self.spectrogram_type not in ("linear", "mel"):
            raise ValueError(
                f"Unsupported spectrogram_type '{self.spectrogram_type}'. "
                "Expected 'linear' or 'mel'."
            )


@dataclass
class AudioInversionConfig:
    """!
    @brief User-facing parameters for normalized spectrogram denormalization and inversion.
    """

    method: str = "gradient"
    gradient_steps: int = 1024
    gradient_lr: float = 0.0005
    gradient_chunk_frames: int = 8192
    gradient_overlap_frames: int = 2048
    decorsiere_alpha: float = 0.3
    decorsiere_lr: float = 1.0
    decorsiere_history_size: int = 10
    min_max_values_path: Optional[str] = None
    use_fixed_db_scale: bool = False
    fixed_min_db: float = -80.0
    fixed_max_db: float = 0.0

    def __post_init__(self) -> None:
        """!
        @brief Normalize scalar fields and validate parameter ranges.
        """
        self.method = str(self.method).strip().lower()
        self.gradient_steps = int(self.gradient_steps)
        self.gradient_lr = float(self.gradient_lr)
        self.gradient_chunk_frames = int(self.gradient_chunk_frames)
        self.gradient_overlap_frames = int(self.gradient_overlap_frames)
        self.decorsiere_alpha = float(self.decorsiere_alpha)
        self.decorsiere_lr = float(self.decorsiere_lr)
        self.decorsiere_history_size = int(self.decorsiere_history_size)
        self.fixed_min_db = float(self.fixed_min_db)
        self.fixed_max_db = float(self.fixed_max_db)
        self.use_fixed_db_scale = bool(self.use_fixed_db_scale)
        self.validate()

    @classmethod
    def from_args(cls, args) -> "AudioInversionConfig":
        """!
        @brief Build config from argparse args while accepting both old and new field names.
        @param args argparse namespace containing audio inversion options.
        @return AudioInversionConfig populated from CLI/default values.
        """
        return cls(
            method=getattr(args, "audio_method", getattr(args, "method", "gradient")),
            gradient_steps=getattr(args, "gradient_inversion_steps", getattr(args, "gradient_steps", 1024)),
            gradient_lr=getattr(args, "gradient_inversion_lr", getattr(args, "gradient_lr", 0.0005)),
            gradient_chunk_frames=getattr(
                args,
                "gradient_inversion_chunk_frames",
                getattr(args, "gradient_chunk_frames", 8192),
            ),
            gradient_overlap_frames=getattr(
                args,
                "gradient_inversion_overlap_frames",
                getattr(args, "gradient_overlap_frames", 2048),
            ),
            decorsiere_alpha=getattr(args, "decorsiere_alpha", 0.3),
            decorsiere_lr=getattr(args, "decorsiere_lr", 1.0),
            decorsiere_history_size=getattr(args, "decorsiere_history_size", 10),
            min_max_values_path=getattr(args, "min_max_values_path", None),
            use_fixed_db_scale=getattr(args, "use_fixed_db_scale", False),
            fixed_min_db=getattr(args, "fixed_min_db", -80.0),
            fixed_max_db=getattr(args, "fixed_max_db", 0.0),
        )

    def validate(self) -> None:
        """!
        @brief Validate inversion parameters before a long generation run starts.
        """
        if self.method not in SUPPORTED_AUDIO_METHODS:
            raise ValueError(f"--audio_method must be one of: {', '.join(SUPPORTED_AUDIO_METHODS)}")
        if self.gradient_steps < 0:
            raise ValueError(f"--gradient_inversion_steps must be >= 0, got {self.gradient_steps}")
        if self.gradient_lr <= 0:
            raise ValueError(f"--gradient_inversion_lr must be > 0, got {self.gradient_lr}")
        if self.gradient_chunk_frames < 8:
            raise ValueError(
                f"--gradient_inversion_chunk_frames must be >= 8, got {self.gradient_chunk_frames}"
            )
        if self.gradient_overlap_frames < 0:
            raise ValueError(
                f"--gradient_inversion_overlap_frames must be >= 0, got {self.gradient_overlap_frames}"
            )
        if self.decorsiere_alpha <= 0:
            raise ValueError(f"--decorsiere_alpha must be > 0, got {self.decorsiere_alpha}")
        if self.decorsiere_lr <= 0:
            raise ValueError(f"--decorsiere_lr must be > 0, got {self.decorsiere_lr}")
        if self.decorsiere_history_size < 1:
            raise ValueError(f"--decorsiere_history_size must be >= 1, got {self.decorsiere_history_size}")
        if self.fixed_max_db <= self.fixed_min_db:
            raise ValueError("--fixed_max_db must be greater than --fixed_min_db")

    def to_legacy_kwargs(self) -> dict:
        """!
        @brief Return kwargs accepted by the previous SoundGenerator API.
        @return Dict of legacy keyword arguments.
        """
        return {
            "method": self.method,
            "gradient_steps": self.gradient_steps,
            "gradient_lr": self.gradient_lr,
            "gradient_chunk_frames": self.gradient_chunk_frames,
            "gradient_overlap_frames": self.gradient_overlap_frames,
            "decorsiere_alpha": self.decorsiere_alpha,
            "decorsiere_lr": self.decorsiere_lr,
            "decorsiere_history_size": self.decorsiere_history_size,
        }

    def to_dict(self) -> dict:
        """!
        @brief Convert this config to a JSON-serializable dictionary.
        @return Dataclass fields as a plain dictionary.
        """
        return asdict(self)


class SpectrogramInverter:
    """!
    @brief Base class for algorithms that invert one denormalized log spectrogram.
    """

    top_db = 80.0

    def __init__(self, geometry: AudioGeometry):
        """!
        @brief Initialize shared inversion state.
        @param geometry STFT/Mel geometry used by the inverter.
        """
        self.geometry = geometry
        self._mel_pinv = None
        self._warning_printed = False

    @property
    def hop_length(self) -> int:
        """!
        @brief Return the STFT hop length.
        @return Hop length in samples.
        """
        return self.geometry.hop_length

    @property
    def sample_rate(self) -> int:
        """!
        @brief Return the audio sample rate.
        @return Sample rate in Hz.
        """
        return self.geometry.sample_rate

    @property
    def n_fft(self) -> int:
        """!
        @brief Return the FFT frame size.
        @return FFT size in samples.
        """
        return self.geometry.n_fft

    @property
    def n_mels(self) -> int:
        """!
        @brief Return the number of Mel bands.
        @return Mel band count.
        """
        return self.geometry.n_mels

    @property
    def spectrogram_type(self) -> str:
        """!
        @brief Return the spectrogram family.
        @return Either `linear` or `mel`.
        """
        return self.geometry.spectrogram_type

    def invert(self, log_spectrogram: np.ndarray) -> np.ndarray:
        """!
        @brief Invert one denormalized log spectrogram.
        @param log_spectrogram 2D spectrogram in dB scale.
        @return Recovered waveform.
        """
        raise NotImplementedError

    def _stft_magnitude_from_log_spectrogram(self, log_spectrogram):
        """!
        @brief Convert a log spectrogram into an STFT magnitude estimate.
        @param log_spectrogram 2D log spectrogram in dB scale.
        @return STFT magnitude matrix.
        """
        log_spectrogram = np.asarray(log_spectrogram, dtype=np.float32)
        if self.spectrogram_type == "mel":
            mel_power = np.power(10.0, log_spectrogram / 10.0)
            mel_power = np.nan_to_num(mel_power, nan=0.0, posinf=1e6, neginf=0.0)
            mel_power = np.maximum(mel_power, 0.0)
            stft_power = self._mel_pseudoinverse().dot(mel_power)
            stft_power = np.nan_to_num(stft_power, nan=0.0, posinf=1e6, neginf=0.0)
            return np.sqrt(np.maximum(stft_power, 0.0)).astype(np.float32, copy=False)

        amplitude = np.power(10.0, log_spectrogram / 20.0)
        amplitude = np.nan_to_num(amplitude, nan=0.0, posinf=1e6, neginf=0.0)
        amplitude = np.maximum(amplitude, 0.0)
        if amplitude.shape[0] == self.n_fft // 2:
            amplitude = np.pad(amplitude, ((0, 1), (0, 0)), mode="constant")
        return amplitude.astype(np.float32, copy=False)

    def _target_power_envelope(self, log_spectrogram):
        """!
        @brief Convert a log spectrogram into a nonnegative power envelope.
        @param log_spectrogram 2D log spectrogram in dB scale.
        @return Power-domain envelope.
        """
        log_spectrogram = np.asarray(log_spectrogram, dtype=np.float32)
        if self.spectrogram_type == "mel":
            power = np.power(10.0, log_spectrogram / 10.0)
        else:
            amplitude = np.power(10.0, log_spectrogram / 20.0)
            power = amplitude * amplitude
        power = np.nan_to_num(power, nan=0.0, posinf=1e6, neginf=0.0)
        return np.maximum(power, 0.0).astype(np.float32, copy=False)

    def _mel_pseudoinverse(self):
        """!
        @brief Return a cached pseudo-inverse of the Mel filterbank.
        @return Pseudo-inverse Mel filter matrix.
        """
        if self._mel_pinv is None:
            self._mel_pinv = np.linalg.pinv(self._mel_filterbank()).astype(np.float32, copy=False)
        return self._mel_pinv

    def _mel_filterbank(self):
        """!
        @brief Build the Slaney-normalized Mel filterbank used in preprocessing.
        @return Mel filterbank matrix.
        """
        try:
            return librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=0.0,
                fmax=float(self.sample_rate) / 2.0,
                htk=False,
                norm="slaney",
            ).astype(np.float32, copy=False)
        except Exception as exc:
            if not self._warning_printed:
                print(f"Warning: librosa Mel filter creation failed; using local Slaney fallback. ({exc})")
                self._warning_printed = True
            return self._fallback_mel_filterbank()

    def _fallback_mel_filterbank(self):
        """!
        @brief Local fallback for librosa's Slaney Mel filterbank.
        @return Mel filterbank matrix.
        """
        fft_frequencies = np.linspace(0.0, self.sample_rate / 2.0, self.n_fft // 2 + 1)
        mel_min = self._hz_to_mel(np.array([0.0]))[0]
        mel_max = self._hz_to_mel(np.array([self.sample_rate / 2.0]))[0]
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        mel_frequencies = self._mel_to_hz(mel_points)

        ramps = mel_frequencies[:, np.newaxis] - fft_frequencies[np.newaxis, :]
        lower = -ramps[:-2] / (mel_frequencies[1:-1] - mel_frequencies[:-2])[:, np.newaxis]
        upper = ramps[2:] / (mel_frequencies[2:] - mel_frequencies[1:-1])[:, np.newaxis]
        weights = np.maximum(0.0, np.minimum(lower, upper))

        enorm = 2.0 / (mel_frequencies[2 : self.n_mels + 2] - mel_frequencies[:self.n_mels])
        weights *= enorm[:, np.newaxis]
        return weights.astype(np.float32, copy=False)

    def _differentiable_log_spectrogram(self, waveform, window, mel_filter=None, target_freq_bins=None):
        """!
        @brief Compute a differentiable log spectrogram from a waveform tensor.
        @param waveform Tensor shaped `(batch, samples)`.
        @param window Hann window tensor for torch.stft.
        @param mel_filter Optional Mel filter tensor for Mel spectrograms.
        @param target_freq_bins Optional frequency-bin count to crop/pad toward.
        @return Log spectrogram tensor.
        """
        pad_mode = "reflect" if waveform.shape[-1] > self.n_fft // 2 else "constant"
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            pad_mode=pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        if self.spectrogram_type == "mel":
            power = stft.abs().pow(2.0)
            mel_power = torch.einsum("mf,bft->bmt", mel_filter, power)
            mel_power = torch.clamp(mel_power, min=1e-10)
            log_spec = 10.0 * torch.log10(mel_power)
        else:
            amplitude = torch.clamp(stft.abs(), min=1e-5)
            if target_freq_bins is not None and amplitude.shape[1] > target_freq_bins:
                amplitude = amplitude[:, :target_freq_bins, :]
            log_spec = 20.0 * torch.log10(amplitude)

        max_per_sample = log_spec.amax(dim=(1, 2), keepdim=True)
        return torch.maximum(log_spec, max_per_sample - self.top_db)

    def _differentiable_power_envelope(self, waveform, window, mel_filter=None, target_freq_bins=None):
        """!
        @brief Compute a differentiable power envelope from a waveform tensor.
        @param waveform Tensor shaped `(batch, samples)`.
        @param window Hann window tensor for torch.stft.
        @param mel_filter Optional Mel filter tensor for Mel spectrograms.
        @param target_freq_bins Optional frequency-bin count to crop toward.
        @return Power-domain spectrogram/envelope tensor.
        """
        pad_mode = "reflect" if waveform.shape[-1] > self.n_fft // 2 else "constant"
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            pad_mode=pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        if self.spectrogram_type == "mel":
            return torch.einsum("mf,bft->bmt", mel_filter, power).clamp_min(1e-12)
        if target_freq_bins is not None and power.shape[1] > target_freq_bins:
            power = power[:, :target_freq_bins, :]
        return power.clamp_min(1e-12)

    @staticmethod
    def _match_spectrogram_shape(spec, target_freq_bins, target_frames):
        """!
        @brief Crop or pad a tensor spectrogram to the requested shape.
        @param spec Tensor shaped `(batch, freq, time)`.
        @param target_freq_bins Desired frequency-bin count.
        @param target_frames Desired time-frame count.
        @return Shape-matched tensor.
        """
        current_bins = int(spec.shape[1])
        if current_bins > target_freq_bins:
            spec = spec[:, :target_freq_bins, :]
        elif current_bins < target_freq_bins:
            spec = F.pad(spec, (0, 0, 0, target_freq_bins - current_bins))

        current_frames = int(spec.shape[-1])
        if current_frames == target_frames:
            return spec
        if current_frames > target_frames:
            return spec[..., :target_frames]
        return F.pad(spec, (0, target_frames - current_frames))

    @staticmethod
    def _match_complex_time_frames(spec, target_frames):
        """!
        @brief Crop or pad a complex STFT to the requested time length.
        @param spec Complex STFT tensor.
        @param target_frames Desired time-frame count.
        @return Time-matched complex STFT tensor.
        """
        current_frames = int(spec.shape[-1])
        if current_frames == target_frames:
            return spec
        if current_frames > target_frames:
            return spec[..., :target_frames]
        return F.pad(spec, (0, target_frames - current_frames))

    @staticmethod
    def _atanh(x):
        """!
        @brief Numerically stable inverse hyperbolic tangent for waveform parameters.
        @param x Tensor constrained to approximately `[-1, 1]`.
        @return atanh(x) tensor.
        """
        x = torch.clamp(x, min=-0.999999, max=0.999999)
        return 0.5 * torch.log((1.0 + x) / (1.0 - x))

    @staticmethod
    def _overlap_add_chunk(output, chunk, start_sample):
        """!
        @brief Insert a waveform chunk into an output buffer with linear crossfade.
        @param output Existing output waveform.
        @param chunk Chunk waveform to add.
        @param start_sample Sample index where the chunk starts.
        @return Updated output waveform.
        """
        start_sample = int(max(0, start_sample))
        chunk = np.asarray(chunk, dtype=np.float32)
        old_length = int(output.shape[0])

        if start_sample > old_length:
            output = np.pad(output, (0, start_sample - old_length), mode="constant")

        end_sample = start_sample + chunk.shape[0]
        if end_sample > output.shape[0]:
            output = np.pad(output, (0, end_sample - output.shape[0]), mode="constant")

        overlap = max(0, min(old_length - start_sample, chunk.shape[0]))
        if overlap > 0:
            fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            output[start_sample:start_sample + overlap] = (
                output[start_sample:start_sample + overlap] * (1.0 - fade)
                + chunk[:overlap] * fade
            )
        if overlap < chunk.shape[0]:
            output[start_sample + overlap:end_sample] = chunk[overlap:]
        return output

    @staticmethod
    def _hz_to_mel(frequencies):
        """!
        @brief Convert frequencies in Hz to Slaney Mel values.
        @param frequencies Scalar or array of frequencies in Hz.
        @return Mel-scale values.
        """
        frequencies = np.asanyarray(frequencies, dtype=np.float64)
        f_sp = 200.0 / 3
        mels = frequencies / f_sp
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
        return mels

    @staticmethod
    def _mel_to_hz(mels):
        """!
        @brief Convert Slaney Mel values to frequencies in Hz.
        @param mels Scalar or array of Mel-scale values.
        @return Frequencies in Hz.
        """
        mels = np.asanyarray(mels, dtype=np.float64)
        f_sp = 200.0 / 3
        frequencies = f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        frequencies[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
        return frequencies


class LibrosaSpectrogramInverter(SpectrogramInverter):
    """!
    @brief Griffin-Lim and direct ISTFT inversion implemented with librosa.
    """

    def __init__(self, geometry: AudioGeometry, method: str):
        """!
        @brief Initialize a librosa-backed inverter.
        @param geometry STFT/Mel geometry.
        @param method Either `griffinlim` or `istft`.
        """
        super().__init__(geometry)
        self.method = str(method).strip().lower()
        if self.method not in ("griffinlim", "istft"):
            raise ValueError(f"LibrosaSpectrogramInverter does not support method '{method}'.")

    def invert(self, log_spectrogram: np.ndarray) -> np.ndarray:
        """!
        @brief Invert one log spectrogram with Griffin-Lim or direct ISTFT.
        @param log_spectrogram 2D spectrogram in dB scale.
        @return Recovered waveform.
        """
        if self.spectrogram_type == "mel":
            mel_power = librosa.db_to_power(log_spectrogram)
            mel_power = np.nan_to_num(mel_power, nan=0.0, posinf=1e6, neginf=0.0)
            mel_power = np.maximum(mel_power, 0.0)

            if self.method == "griffinlim":
                audio_signal = librosa.feature.inverse.mel_to_audio(
                    M=mel_power,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    power=2.0,
                )
            else:
                magnitude_stft = librosa.feature.inverse.mel_to_stft(
                    M=mel_power,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    power=2.0,
                )
                audio_signal = librosa.istft(magnitude_stft, hop_length=self.hop_length, n_fft=self.n_fft)
        else:
            amplitude_spectrogram = librosa.db_to_amplitude(log_spectrogram)
            amplitude_spectrogram = np.nan_to_num(amplitude_spectrogram, nan=0.0, posinf=1e6, neginf=0.0)
            amplitude_spectrogram = np.maximum(amplitude_spectrogram, 0.0)
            if amplitude_spectrogram.shape[0] == self.n_fft // 2:
                amplitude_spectrogram = np.pad(amplitude_spectrogram, ((0, 1), (0, 0)), mode="constant")

            if self.method == "griffinlim":
                audio_signal = librosa.griffinlim(
                    amplitude_spectrogram,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft,
                )
            else:
                audio_signal = librosa.istft(
                    amplitude_spectrogram,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft,
                )

        return np.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)


class GradientSpectrogramInverter(SpectrogramInverter):
    """!
    @brief Adam optimization of a waveform to match the target log spectrogram.
    """

    def __init__(self, geometry: AudioGeometry, config: AudioInversionConfig):
        """!
        @brief Initialize the Adam-based waveform optimizer.
        @param geometry STFT/Mel geometry.
        @param config Inversion hyperparameters.
        """
        super().__init__(geometry)
        self.config = config

    def invert(self, log_spectrogram: np.ndarray) -> np.ndarray:
        """!
        @brief Invert one log spectrogram by optimizing waveform samples.
        @param log_spectrogram 2D spectrogram in dB scale.
        @return Recovered waveform.
        """
        steps = int(self.config.gradient_steps)
        if steps <= 0:
            return LibrosaSpectrogramInverter(self.geometry, "griffinlim").invert(log_spectrogram)

        total_frames = int(log_spectrogram.shape[1])
        if total_frames <= 1:
            return np.zeros(max(1, total_frames * self.hop_length), dtype=np.float32)

        starts, chunk_frames, overlap_frames = self._chunk_starts(total_frames)
        if len(starts) > 1:
            print(
                "Gradient inversion: "
                f"{len(starts)} chunks, {steps} steps/chunk, "
                f"chunk_frames={chunk_frames}, overlap_frames={overlap_frames}"
            )

        output = np.zeros(0, dtype=np.float32)
        for start_frame in starts:
            end_frame = min(total_frames, start_frame + chunk_frames)
            chunk = log_spectrogram[:, start_frame:end_frame]
            chunk_audio = self._optimize_chunk(chunk)
            output = self._overlap_add_chunk(output, chunk_audio, start_sample=start_frame * self.hop_length)

        target_length = max(1, (total_frames - 1) * self.hop_length)
        if output.shape[0] < target_length:
            output = np.pad(output, (0, target_length - output.shape[0]), mode="constant")
        else:
            output = output[:target_length]
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _chunk_starts(self, total_frames: int):
        """!
        @brief Compute frame starts for chunked gradient inversion.
        @param total_frames Total spectrogram time frames.
        @return Tuple `(starts, chunk_frames, overlap_frames)`.
        """
        chunk_frames = max(8, int(self.config.gradient_chunk_frames))
        overlap_frames = max(0, int(self.config.gradient_overlap_frames))
        overlap_frames = min(overlap_frames, max(0, chunk_frames - 1))
        step_frames = max(1, chunk_frames - overlap_frames)
        last_start = max(0, total_frames - chunk_frames)
        starts = list(range(0, last_start + 1, step_frames))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts, chunk_frames, overlap_frames

    def _optimize_chunk(self, log_spectrogram):
        """!
        @brief Optimize one spectrogram chunk in the time domain.
        @param log_spectrogram Chunk log spectrogram in dB scale.
        @return Recovered waveform chunk.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = torch.from_numpy(np.asarray(log_spectrogram, dtype=np.float32)).to(device)
        target = torch.nan_to_num(target, nan=0.0, posinf=80.0, neginf=-120.0).unsqueeze(0)
        target_freq_bins = int(target.shape[1])
        target_frames = int(target.shape[-1])
        target_length = max(1, (target_frames - 1) * self.hop_length)

        init_audio = self._initial_audio(log_spectrogram, target_length)
        if init_audio.shape[0] < target_length:
            init_audio = np.pad(init_audio, (0, target_length - init_audio.shape[0]), mode="constant")
        else:
            init_audio = init_audio[:target_length]

        init = torch.from_numpy(init_audio).to(device)
        init = torch.clamp(init, min=-0.95, max=0.95)
        raw_waveform = torch.nn.Parameter(self._atanh(init))
        optimizer = torch.optim.Adam([raw_waveform], lr=float(self.config.gradient_lr))

        window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32, device=device)
        mel_filter = None
        if self.spectrogram_type == "mel":
            mel_filter = torch.from_numpy(self._mel_filterbank()).to(device=device, dtype=torch.float32)

        best_loss = float("inf")
        best_waveform = init.detach().clone()
        stale_steps = 0
        patience = max(8, min(32, int(self.config.gradient_steps) // 4))

        with torch.enable_grad():
            for _ in range(int(self.config.gradient_steps)):
                waveform = torch.tanh(raw_waveform).unsqueeze(0)
                pred = self._differentiable_log_spectrogram(
                    waveform,
                    window=window,
                    mel_filter=mel_filter,
                    target_freq_bins=target_freq_bins,
                )
                pred = self._match_spectrogram_shape(pred, target_freq_bins, target_frames)
                loss = F.l1_loss(pred, target)
                loss_value = float(loss.detach().cpu().item())
                if loss_value < best_loss - 1e-4:
                    best_loss = loss_value
                    best_waveform = waveform.detach().squeeze(0).clone()
                    stale_steps = 0
                else:
                    stale_steps += 1
                    if stale_steps >= patience:
                        break

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        audio = best_waveform.detach().cpu().numpy()
        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _initial_audio(self, log_spectrogram, target_length):
        """!
        @brief Build the initial waveform for Adam optimization.
        @param log_spectrogram Target chunk log spectrogram.
        @param target_length Desired waveform length in samples.
        @return Initial waveform.
        """
        try:
            init_audio = self._torch_griffinlim_initialization(log_spectrogram, target_length=target_length, n_iter=32)
        except Exception as exc:
            if not self._warning_printed:
                print(f"Warning: Griffin-Lim initialization failed for gradient inversion; using quiet noise. ({exc})")
                self._warning_printed = True
            rng = np.random.default_rng(0)
            init_audio = rng.normal(0.0, 1e-4, size=target_length)
        return np.nan_to_num(np.asarray(init_audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _torch_griffinlim_initialization(self, log_spectrogram, target_length, n_iter=32):
        """!
        @brief Run torch Griffin-Lim to initialize gradient inversion.
        @param log_spectrogram Target log spectrogram in dB scale.
        @param target_length Desired waveform length in samples.
        @param n_iter Number of phase-reconstruction iterations.
        @return Initial waveform.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        magnitude = self._stft_magnitude_from_log_spectrogram(log_spectrogram)
        magnitude = torch.from_numpy(magnitude).to(device=device, dtype=torch.float32)
        window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32, device=device)

        phase = torch.rand_like(magnitude) * (2.0 * np.pi) - np.pi
        complex_spec = torch.polar(magnitude, phase)
        for _ in range(int(n_iter)):
            audio = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                length=int(target_length),
            )
            rebuilt = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                center=True,
                pad_mode="reflect" if audio.shape[-1] > self.n_fft // 2 else "constant",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            rebuilt = self._match_complex_time_frames(rebuilt, magnitude.shape[-1])
            complex_spec = torch.polar(magnitude, torch.angle(rebuilt))

        audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=int(target_length),
        )
        return audio.detach().cpu().numpy()


class DecorsiereSpectrogramInverter(SpectrogramInverter):
    """!
    @brief L-BFGS time-domain optimization inspired by Decorsiere et al. (2015).
    """

    def __init__(self, geometry: AudioGeometry, config: AudioInversionConfig):
        """!
        @brief Initialize the Decorsiere-style L-BFGS optimizer.
        @param geometry STFT/Mel geometry.
        @param config Inversion hyperparameters.
        """
        super().__init__(geometry)
        self.config = config

    def invert(self, log_spectrogram: np.ndarray) -> np.ndarray:
        """!
        @brief Invert one log spectrogram using compressed-envelope L-BFGS optimization.
        @param log_spectrogram 2D spectrogram in dB scale.
        @return Recovered waveform.
        """
        max_iter = int(self.config.gradient_steps)
        if max_iter <= 0:
            return LibrosaSpectrogramInverter(self.geometry, "griffinlim").invert(log_spectrogram)

        total_frames = int(log_spectrogram.shape[1])
        if total_frames <= 1:
            return np.zeros(max(1, total_frames * self.hop_length), dtype=np.float32)

        starts, chunk_frames, overlap_frames = self._chunk_starts(total_frames)
        if len(starts) > 1:
            print(
                "Decorsiere-style inversion: "
                f"{len(starts)} chunks, {max_iter} L-BFGS iterations/chunk, "
                f"chunk_frames={chunk_frames}, overlap_frames={overlap_frames}, "
                f"alpha={self.config.decorsiere_alpha}"
            )

        output = np.zeros(0, dtype=np.float32)
        for start_frame in starts:
            end_frame = min(total_frames, start_frame + chunk_frames)
            chunk = log_spectrogram[:, start_frame:end_frame]
            chunk_audio = self._optimize_chunk(chunk)
            output = self._overlap_add_chunk(output, chunk_audio, start_sample=start_frame * self.hop_length)

        target_length = max(1, (total_frames - 1) * self.hop_length)
        if output.shape[0] < target_length:
            output = np.pad(output, (0, target_length - output.shape[0]), mode="constant")
        else:
            output = output[:target_length]
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _chunk_starts(self, total_frames: int):
        """!
        @brief Compute frame starts for chunked Decorsiere-style inversion.
        @param total_frames Total spectrogram time frames.
        @return Tuple `(starts, chunk_frames, overlap_frames)`.
        """
        chunk_frames = max(8, int(self.config.gradient_chunk_frames))
        overlap_frames = max(0, int(self.config.gradient_overlap_frames))
        overlap_frames = min(overlap_frames, max(0, chunk_frames - 1))
        step_frames = max(1, chunk_frames - overlap_frames)
        last_start = max(0, total_frames - chunk_frames)
        starts = list(range(0, last_start + 1, step_frames))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts, chunk_frames, overlap_frames

    def _optimize_chunk(self, log_spectrogram):
        """!
        @brief Optimize one chunk with the Decorsiere-style objective.
        @param log_spectrogram Chunk log spectrogram in dB scale.
        @return Recovered waveform chunk.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_power = self._target_power_envelope(log_spectrogram)
        target = torch.from_numpy(target_power).to(device=device, dtype=torch.float32).unsqueeze(0)
        target_freq_bins = int(target.shape[1])
        target_frames = int(target.shape[-1])
        target_length = max(1, (target_frames - 1) * self.hop_length)

        init_audio = self._random_phase_initialization(log_spectrogram, target_length)
        if init_audio.shape[0] < target_length:
            init_audio = np.pad(init_audio, (0, target_length - init_audio.shape[0]), mode="constant")
        else:
            init_audio = init_audio[:target_length]

        waveform = torch.nn.Parameter(torch.from_numpy(init_audio).to(device=device, dtype=torch.float32))
        window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32, device=device)
        mel_filter = None
        if self.spectrogram_type == "mel":
            mel_filter = torch.from_numpy(self._mel_filterbank()).to(device=device, dtype=torch.float32)

        target_compressed = torch.clamp(target, min=1e-12).pow(float(self.config.decorsiere_alpha))
        normalizer = torch.clamp(torch.mean(target_compressed.pow(2.0)), min=1e-12)
        best_loss = float("inf")
        best_waveform = waveform.detach().clone()

        optimizer = torch.optim.LBFGS(
            [waveform],
            lr=float(self.config.decorsiere_lr),
            max_iter=int(self.config.gradient_steps),
            max_eval=max(int(self.config.gradient_steps) * 2, int(self.config.gradient_steps) + 1),
            history_size=max(1, int(self.config.decorsiere_history_size)),
            line_search_fn="strong_wolfe",
        )

        def closure():
            """!
            @brief L-BFGS closure that computes loss and gradients.
            @return Scalar objective tensor.
            """
            nonlocal best_loss, best_waveform
            optimizer.zero_grad(set_to_none=True)
            pred_power = self._differentiable_power_envelope(
                waveform.unsqueeze(0),
                window=window,
                mel_filter=mel_filter,
                target_freq_bins=target_freq_bins,
            )
            pred_power = self._match_spectrogram_shape(pred_power, target_freq_bins, target_frames)
            pred_compressed = torch.clamp(pred_power, min=1e-12).pow(float(self.config.decorsiere_alpha))
            loss = torch.mean((pred_compressed - target_compressed).pow(2.0)) / normalizer
            loss_value = float(loss.detach().cpu().item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_waveform = waveform.detach().clone()
            loss.backward()
            return loss

        with torch.enable_grad():
            optimizer.step(closure)

        return np.nan_to_num(
            best_waveform.detach().cpu().numpy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

    def _random_phase_initialization(self, log_spectrogram, target_length):
        """!
        @brief Initialize a waveform from random STFT phase.
        @param log_spectrogram Target log spectrogram in dB scale.
        @param target_length Desired waveform length in samples.
        @return Initial waveform.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        magnitude = self._stft_magnitude_from_log_spectrogram(log_spectrogram)
        magnitude = torch.from_numpy(magnitude).to(device=device, dtype=torch.float32)
        phase = torch.rand_like(magnitude) * (2.0 * np.pi) - np.pi
        complex_spec = torch.polar(magnitude, phase)
        window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32, device=device)
        audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=int(target_length),
        )
        return audio.detach().cpu().numpy()


def build_spectrogram_inverter(
    config: AudioInversionConfig,
    geometry: AudioGeometry,
) -> SpectrogramInverter:
    """!
    @brief Factory for the spectrogram inversion method selected by config.
    @param config Audio inversion settings.
    @param geometry STFT/Mel geometry.
    @return Concrete SpectrogramInverter instance.
    """
    if config.method in ("griffinlim", "istft"):
        return LibrosaSpectrogramInverter(geometry, config.method)
    if config.method == "gradient":
        return GradientSpectrogramInverter(geometry, config)
    if config.method == "decorsiere":
        return DecorsiereSpectrogramInverter(geometry, config)
    raise ValueError(f"Unsupported inversion method: {config.method}")
