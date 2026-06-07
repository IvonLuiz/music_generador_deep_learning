from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from generation.audio_inversion import AudioInversionConfig


@dataclass
class PriorPathsConfig:
    """!
    @brief Model/config paths used by hierarchical music generation.
    """

    top_config: Optional[str]
    middle_config: Optional[str]
    bottom_config: Optional[str]
    top_run_root: str
    middle_run_root: str
    bottom_run_root: str
    weights_file: str = "best_model.pth"


@dataclass
class SamplingConfig:
    """!
    @brief Autoregressive sampling controls.
    """

    temperature: float = 1.0
    top_k: Optional[int] = None
    duration_seconds: float = 30.0
    seed: Optional[int] = 42


@dataclass
class WindowingConfig:
    """!
    @brief Window placement, overlap, and bottom decoding controls.
    """

    sampling_mode: str = "windowed"
    overlap_fraction: float = 0.5
    windowed_prefix_levels: str = "all"
    bottom_decode_mode: str = "timeline"
    bottom_decode_context_cols: int = -1


@dataclass
class OutputConfig:
    """!
    @brief Output directory and optional artifact flags.
    """

    save_root: str = "samples/generate_music_maestro"
    save_middle_audio: bool = True


@dataclass
class GenerationConfig:
    """!
    @brief Structured parameter bundle for generate_music.py.
    """

    priors: PriorPathsConfig
    sampling: SamplingConfig
    windowing: WindowingConfig
    audio: AudioInversionConfig
    output: OutputConfig

    @classmethod
    def from_args(cls, args) -> "GenerationConfig":
        """!
        @brief Build structured generation settings from argparse output.
        @param args argparse namespace from generate_music.py.
        @return Validated GenerationConfig.
        """
        top_k = getattr(args, "top_k", None)
        if top_k is not None and top_k < 0:
            raise ValueError(f"--top_k must be >= 0, got {top_k}")
        top_k = top_k if top_k is not None and top_k > 0 else None
        seed = getattr(args, "seed", 42)
        seed = None if seed is not None and seed < 0 else seed
        weights_file = getattr(args, "weights_file", "best_model.pth")
        if not weights_file.endswith(".pth"):
            weights_file += ".pth"

        config = cls(
            priors=PriorPathsConfig(
                top_config=getattr(args, "top_config", None),
                middle_config=getattr(args, "middle_config", None),
                bottom_config=getattr(args, "bottom_config", None),
                top_run_root=getattr(args, "top_run_root"),
                middle_run_root=getattr(args, "middle_run_root"),
                bottom_run_root=getattr(args, "bottom_run_root"),
                weights_file=weights_file,
            ),
            sampling=SamplingConfig(
                temperature=float(getattr(args, "temperature", 1.0)),
                top_k=top_k,
                duration_seconds=float(getattr(args, "duration_seconds", 30.0)),
                seed=seed,
            ),
            windowing=WindowingConfig(
                sampling_mode=getattr(args, "sampling_mode", "windowed"),
                overlap_fraction=float(getattr(args, "overlap_fraction", 0.5)),
                windowed_prefix_levels=getattr(args, "windowed_prefix_levels", "all"),
                bottom_decode_mode=getattr(args, "bottom_decode_mode", "timeline"),
                bottom_decode_context_cols=int(getattr(args, "bottom_decode_context_cols", -1)),
            ),
            audio=AudioInversionConfig.from_args(args),
            output=OutputConfig(
                save_root=getattr(args, "save_root", "samples/generate_music_maestro"),
                save_middle_audio=bool(getattr(args, "save_middle_audio", True)),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """!
        @brief Validate generation settings before loading large models.
        """
        if self.sampling.temperature <= 0:
            raise ValueError(f"--temperature must be > 0, got {self.sampling.temperature}")
        if self.sampling.duration_seconds <= 0:
            raise ValueError(f"--duration_seconds must be > 0, got {self.sampling.duration_seconds}")
        if not 0.0 <= self.windowing.overlap_fraction < 1.0:
            raise ValueError(f"--overlap_fraction must be in [0, 1), got {self.windowing.overlap_fraction}")
        if self.windowing.bottom_decode_context_cols < -1:
            raise ValueError(
                f"--bottom_decode_context_cols must be >= -1, got {self.windowing.bottom_decode_context_cols}"
            )
        self.audio.validate()

    def apply_to_args(self, args) -> None:
        """!
        @brief Keep legacy args-based code normalized while the script is migrated.
        @param args argparse namespace to update in-place.
        """
        args.top_k = self.sampling.top_k
        args.seed = self.sampling.seed
        args.weights_file = self.priors.weights_file
        args.audio_method = self.audio.method
        args.generation_config = self
        args.audio_inversion_config = self.audio

    def to_dict(self) -> dict:
        """!
        @brief Convert this config to a JSON-serializable dictionary.
        @return Nested dataclass fields as a plain dictionary.
        """
        return asdict(self)
