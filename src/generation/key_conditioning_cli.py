from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from metadata.title_key import key_metadata_from_id, key_metadata_from_label


@dataclass(frozen=True)
class KeyConditioningConfig:
    """!
    @brief Resolved key-conditioning values passed to Transformer priors.
    """

    key_id: int
    key_label: str
    key_source: str

    @classmethod
    def from_metadata(cls, metadata: dict) -> "KeyConditioningConfig":
        """!
        @brief Build a key-conditioning config from metadata.title_key output.
        @param metadata Dictionary containing `key_id`, `key_label`, and `key_source`.
        @return Resolved key-conditioning config.
        """
        return cls(
            key_id=int(metadata["key_id"]),
            key_label=str(metadata["key_label"]),
            key_source=str(metadata["key_source"]),
        )

    def to_dict(self) -> dict:
        """!
        @brief Convert this config to a JSON-serializable dictionary.
        @return Plain dictionary with key-conditioning fields.
        """
        return {
            "key_id": int(self.key_id),
            "key_label": self.key_label,
            "key_source": self.key_source,
        }


def add_key_conditioning_args(parser: argparse.ArgumentParser) -> None:
    """!
    @brief Add optional key-conditioning CLI arguments to a parser.
    @param parser Parser to update in-place.
    """
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help=(
            "Optional Transformer key conditioning label, e.g. "
            "'C major', 'A minor', 'F# minor', 'Bb major', or 'unknown'."
        ),
    )
    parser.add_argument(
        "--key_id",
        type=int,
        default=None,
        help="Optional raw Transformer key class ID. Valid IDs are 0-24; 24 means unknown.",
    )


def resolve_key_conditioning(
    key_label: Optional[str],
    key_id: Optional[int],
) -> Optional[KeyConditioningConfig]:
    """!
    @brief Resolve CLI key-conditioning arguments into a single config object.
    @param key_label Optional human-readable key label.
    @param key_id Optional raw key class ID.
    @return Resolved key config, or None when no key conditioning was requested.
    """
    has_label = key_label is not None and str(key_label).strip() != ""
    has_id = key_id is not None
    if has_label and has_id:
        raise ValueError("Use either --key or --key_id, not both.")
    if has_id:
        return KeyConditioningConfig.from_metadata(key_metadata_from_id(int(key_id), source="cli_id"))
    if has_label:
        return KeyConditioningConfig.from_metadata(key_metadata_from_label(str(key_label)))
    return None
