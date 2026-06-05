from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SpectrogramPlotConfig:
    """!
    @brief Shared spectrogram plot styling.
    """

    cmap: str = "magma"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    colorbar_label: str = "Normalized amplitude"
    x_label: str = "Time frames"
    y_label: str = "Frequency bins"
    dpi: int = 150


class SpectrogramVisualizer:
    """!
    @brief Shared plotting helper for test/evaluation scripts.
    """

    def __init__(self, config: Optional[SpectrogramPlotConfig] = None):
        """!
        @brief Initialize plot style.
        @param config Optional plot config.
        """
        self.config = config or SpectrogramPlotConfig()

    @staticmethod
    def as_2d(spec: np.ndarray) -> np.ndarray:
        """!
        @brief Normalize spectrogram arrays to `(freq, time)`.
        @param spec Spectrogram shaped `(F, T)` or `(F, T, 1)`.
        @return 2D spectrogram.
        """
        return spec[:, :, 0] if spec.ndim == 3 else spec

    def save_spectrogram(self, spec: np.ndarray, out_path: str, title: str, figsize=(10, 4)) -> str:
        """!
        @brief Save one spectrogram image with consistent axis labels.
        @param spec Spectrogram shaped `(F, T)` or `(F, T, 1)`.
        @param out_path Output image path.
        @param title Plot title.
        @param figsize Figure size in inches.
        @return Written image path.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img = self.as_2d(spec)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            cmap=self.config.cmap,
            vmin=self.config.vmin,
            vmax=self.config.vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(self.config.x_label)
        ax.set_ylabel(self.config.y_label)
        fig.colorbar(im, ax=ax, label=self.config.colorbar_label, pad=0.015)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def save_batch(self, specs: np.ndarray, out_dir: str, prefix: str, title_prefix: str) -> List[str]:
        """!
        @brief Save a batch of spectrogram images and its `.npy` array.
        @param specs Spectrogram batch shaped `(B, F, T, 1)` or `(B, F, T)`.
        @param out_dir Output directory.
        @param prefix File name prefix.
        @param title_prefix Plot title prefix.
        @return List of written image paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{prefix}.npy"), specs)
        paths: List[str] = []
        for idx in range(specs.shape[0]):
            path = os.path.join(out_dir, f"{prefix}_{idx:03d}.png")
            self.save_spectrogram(specs[idx], path, f"{title_prefix} {idx}")
            paths.append(path)
        return paths

    def save_comparisons(self, original: np.ndarray, reconstructed: np.ndarray, out_dir: str) -> List[str]:
        """!
        @brief Save original/reconstructed/error comparison plots.
        @param original Original spectrogram batch.
        @param reconstructed Reconstructed spectrogram batch.
        @param out_dir Output directory.
        @return List of written image paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "original_specs.npy"), original)
        np.save(os.path.join(out_dir, "reconstructed_specs.npy"), reconstructed)
        paths: List[str] = []
        count = min(original.shape[0], reconstructed.shape[0])
        for idx in range(count):
            orig = self.as_2d(original[idx])
            recon = self.as_2d(reconstructed[idx])
            min_t = min(orig.shape[1], recon.shape[1])
            orig = orig[:, :min_t]
            recon = recon[:, :min_t]
            diff = np.abs(orig - recon)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            panels = [
                ("Original", orig, self.config.cmap, self.config.vmin, self.config.vmax, self.config.colorbar_label),
                ("Reconstructed", recon, self.config.cmap, self.config.vmin, self.config.vmax, self.config.colorbar_label),
                ("Absolute error", diff, "hot", 0.0, 0.4, "|Original - Reconstructed|"),
            ]
            for ax, (title, img, cmap, vmin, vmax, label) in zip(axes, panels):
                im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.set_xlabel(self.config.x_label)
                ax.set_ylabel(self.config.y_label)
                fig.colorbar(im, ax=ax, label=label, pad=0.015)
            mse = float(np.mean(diff ** 2))
            mae = float(np.mean(diff))
            fig.suptitle(f"Sample {idx} - MSE: {mse:.6f}, MAE: {mae:.6f}", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            path = os.path.join(out_dir, f"comparison_{idx:03d}.png")
            fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        return paths

    def save_code_indices(self, indices: np.ndarray, out_dir: str, prefix: str = "indices") -> List[str]:
        """!
        @brief Save code-index arrays and per-sample visualizations.
        @param indices Code indices shaped `(B, H, W)` or `(B, T)`.
        @param out_dir Output directory.
        @param prefix File name prefix.
        @return List of written image paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        arr = np.asarray(indices, dtype=np.int64)
        np.save(os.path.join(out_dir, f"{prefix}.npy"), arr)
        paths: List[str] = []
        for idx in range(arr.shape[0]):
            img = arr[idx]
            if img.ndim == 1:
                img = img.reshape(1, -1)
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(img, origin="lower", aspect="auto")
            ax.set_title(f"Code indices {idx}")
            ax.set_xlabel(self.config.x_label)
            ax.set_ylabel(self.config.y_label)
            fig.colorbar(im, ax=ax, label="Code index", pad=0.015)
            fig.tight_layout()
            path = os.path.join(out_dir, f"{prefix}_{idx:03d}.png")
            fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        return paths

    def save_code_histogram(self, indices: np.ndarray, num_embeddings: int, out_path: str) -> str:
        """!
        @brief Save a codebook usage histogram.
        @param indices Code-index array.
        @param num_embeddings Number of codebook entries.
        @param out_path Output image path.
        @return Written image path.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        flat = np.asarray(indices, dtype=np.int64).reshape(-1)
        hist = np.bincount(flat, minlength=int(num_embeddings))
        np.save(os.path.splitext(out_path)[0] + ".npy", hist)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(np.arange(int(num_embeddings)), hist, width=1.0)
        ax.set_title("Codebook Usage Histogram")
        ax.set_xlabel("Code index")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path
