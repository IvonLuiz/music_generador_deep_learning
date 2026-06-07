import csv
import json
import os
import re
from collections import Counter
from typing import Dict, Optional


UNKNOWN_KEY_ID = 24
UNKNOWN_KEY_LABEL = 'unknown'

PITCH_CLASS_TO_LABEL = {
    0: 'C',
    1: 'C-sharp',
    2: 'D',
    3: 'E-flat',
    4: 'E',
    5: 'F',
    6: 'F-sharp',
    7: 'G',
    8: 'A-flat',
    9: 'A',
    10: 'B-flat',
    11: 'B',
}

ROOT_TO_PITCH_CLASS = {
    'c': 0,
    'b#': 0,
    'c#': 1,
    'db': 1,
    'd': 2,
    'd#': 3,
    'eb': 3,
    'e': 4,
    'fb': 4,
    'e#': 5,
    'f': 5,
    'f#': 6,
    'gb': 6,
    'g': 7,
    'g#': 8,
    'ab': 8,
    'a': 9,
    'a#': 10,
    'bb': 10,
    'b': 11,
    'cb': 11,
}


def unknown_key_metadata(source: str = 'unknown') -> dict:
    return {
        'key_id': UNKNOWN_KEY_ID,
        'key_label': UNKNOWN_KEY_LABEL,
        'key_source': source,
    }


def key_id_from_pitch_class(pitch_class: int, mode: str) -> int:
    mode = str(mode).strip().lower()
    if mode not in ('major', 'minor'):
        raise ValueError(f"mode must be 'major' or 'minor', got {mode!r}")
    return int(pitch_class) * 2 + (1 if mode == 'minor' else 0)


def _normalize_title(title: str) -> str:
    text = str(title or '').replace('♭', ' flat ').replace('♯', ' sharp ')
    text = text.replace('–', '-').replace('—', '-').replace('_', ' ')
    text = re.sub(r'(?i)\b([A-G])\s*-\s*(flat|sharp)\b', r'\1 \2', text)
    text = re.sub(r'(?i)\b([A-G])\s+(flat|sharp)\b', r'\1 \2', text)
    text = re.sub(r'(?i)\b([A-G])\s*#\b', r'\1#', text)
    text = re.sub(r'(?i)\b([A-G])\s*b\b', r'\1b', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _canonical_root_token(root: str, accidental: Optional[str]) -> Optional[str]:
    root = str(root or '').strip().lower()
    accidental = str(accidental or '').strip().lower()
    accidental = {
        'flat': 'b',
        'sharp': '#',
        '-flat': 'b',
        '-sharp': '#',
    }.get(accidental, accidental)
    token = f'{root}{accidental}'
    return token if token in ROOT_TO_PITCH_CLASS else None


def _key_metadata(root_token: str, mode: str, source: str) -> dict:
    pitch_class = ROOT_TO_PITCH_CLASS[root_token]
    mode = 'minor' if str(mode).strip().lower().startswith('min') else 'major'
    label = f'{PITCH_CLASS_TO_LABEL[pitch_class]} {mode}'
    return {
        'key_id': key_id_from_pitch_class(pitch_class, mode),
        'key_label': label,
        'key_source': source,
    }


def infer_key_from_title(title: str, infer_missing_mode_as: str = 'major') -> dict:
    """Infer a coarse home-key class from a MAESTRO canonical_title string."""
    text = _normalize_title(title)
    if not text:
        return unknown_key_metadata()

    explicit_pattern = re.compile(
        r'(?i)(?:^|[\s,;:(])(?:in\s+)?'
        r'([A-G])\s*(#|b|flat|sharp)?\s+'
        r'(major|minor|maj\.?|min\.?)\b'
    )
    matches = list(explicit_pattern.finditer(text))
    if matches:
        # Prefer the last explicit title key. This avoids early catalog numbers
        # and handles titles like "Sonata No. 17 ... in D Minor".
        match = matches[-1]
        root_token = _canonical_root_token(match.group(1), match.group(2))
        if root_token is not None:
            return _key_metadata(root_token, match.group(3), 'title_explicit')

    mode = str(infer_missing_mode_as or '').strip().lower()
    if mode in ('major', 'minor'):
        implicit_pattern = re.compile(
            r'(?i)(?:^|[\s,;:(])(?:in\s+)?([A-G])\s*(#|b|flat|sharp)\b'
        )
        matches = list(implicit_pattern.finditer(text))
        if matches:
            match = matches[-1]
            root_token = _canonical_root_token(match.group(1), match.group(2))
            if root_token is not None:
                return _key_metadata(root_token, mode, f'title_implicit_{mode}')

    return unknown_key_metadata()


def read_maestro_rows(metadata_path: Optional[str]) -> list:
    if not metadata_path:
        return []
    metadata_path = os.path.abspath(os.path.expanduser(metadata_path))
    if not os.path.isfile(metadata_path):
        return []
    if metadata_path.lower().endswith('.json'):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [dict(row) for row in data if isinstance(row, dict)]
        if isinstance(data, dict) and all(isinstance(value, dict) for value in data.values()):
            keys = sorted(data.keys(), key=lambda item: int(item) if str(item).isdigit() else str(item))
            row_count = max((len(value) for value in data.values()), default=0)
            rows = []
            for idx in range(row_count):
                row = {}
                for column in keys:
                    values = data[column]
                    row[column] = values.get(str(idx), values.get(idx, ''))
                rows.append(row)
            return rows
        return []
    with open(metadata_path, newline='', encoding='utf-8') as f:
        return [dict(row) for row in csv.DictReader(f)]


def _processed_audio_basename(file_path: str) -> str:
    basename = os.path.basename(file_path)
    if basename.endswith('.npy'):
        basename = basename[:-4]
    match = re.match(r'^(.+)_segment_\d+$', basename)
    if match:
        basename = match.group(1)
    return basename


def build_title_key_metadata_by_source(
    metadata_path: Optional[str],
    infer_missing_mode_as: str = 'major',
) -> Dict[str, dict]:
    """Build lookup entries keyed by basename/stem/relative filename variants."""
    rows = read_maestro_rows(metadata_path)
    by_source: Dict[str, dict] = {}
    for row in rows:
        title = row.get('canonical_title', '')
        inferred = infer_key_from_title(title, infer_missing_mode_as=infer_missing_mode_as)
        entry = {
            **inferred,
            'canonical_title': title,
            'canonical_composer': row.get('canonical_composer', ''),
        }
        for filename_key in ('audio_filename', 'midi_filename'):
            filename = str(row.get(filename_key, '') or '').replace('\\', '/')
            if not filename:
                continue
            basename = os.path.basename(filename)
            stem = os.path.splitext(basename)[0]
            by_source[os.path.normpath(filename)] = entry
            by_source[basename] = entry
            by_source[stem] = entry
    return by_source


def key_metadata_for_path(file_path: str, metadata_by_source: Dict[str, dict]) -> dict:
    if not metadata_by_source:
        return unknown_key_metadata()
    normalized_path = os.path.normpath(str(file_path).replace('\\', '/'))
    candidates = [
        normalized_path,
        os.path.basename(normalized_path),
        os.path.splitext(os.path.basename(normalized_path))[0],
        _processed_audio_basename(normalized_path),
    ]
    for candidate in candidates:
        if candidate in metadata_by_source:
            return dict(metadata_by_source[candidate])
    return unknown_key_metadata()


def key_metadata_counts(metadata_by_source: Dict[str, dict]) -> Dict[str, int]:
    counts = Counter()
    seen = set()
    for source, metadata in metadata_by_source.items():
        key = (metadata.get('canonical_title'), metadata.get('key_id'))
        if key in seen:
            continue
        seen.add(key)
        counts[str(metadata.get('key_label', UNKNOWN_KEY_LABEL))] += 1
    return dict(sorted(counts.items()))
