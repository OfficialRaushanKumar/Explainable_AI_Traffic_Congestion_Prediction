"""
preprocessing.py
================
CCTV Video Preprocessing Pipeline

Responsibilities:
  - Open video files with OpenCV
  - Extract frames at configurable intervals
  - Apply background subtraction (MOG2) to isolate moving vehicles
  - Resize and normalise frames
  - Package frames into fixed-length sequences for the CNN-LSTM model
  - Save processed sequences to disk as NumPy arrays
"""
"""
preprocessing.py
=================
Video discovery, metadata extraction, and sequence preparation.

This module intentionally refuses to fabricate labels. It will scan the raw
dataset, resolve labels from either a manifest or a filename convention, write
an inspection report, and only build training arrays when valid labels exist.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class VideoRecord:
    path: str
    label: Optional[str]
    label_source: str
    readable: bool
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    size_mb: float
    skipped_reason: Optional[str] = None


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class BackgroundSubtractor:
    """Small wrapper around OpenCV MOG2 background subtraction."""

    def __init__(self, history: int = 200, var_threshold: float = 50.0, detect_shadows: bool = False):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.subtractor.apply(frame)
        fg_mask_3ch = cv2.merge([fg_mask, fg_mask, fg_mask])
        return cv2.bitwise_and(frame, fg_mask_3ch)


class FrameExtractor:
    """Extract fixed-length frame sequences from a video."""

    def __init__(
        self,
        height: int = 224,
        width: int = 224,
        sequence_length: int = 16,
        frame_skip: int = 2,
        use_bg_sub: bool = True,
        normalize: bool = True,
    ):
        self.height = height
        self.width = width
        self.sequence_length = sequence_length
        self.frame_skip = max(1, frame_skip)
        self.normalize = normalize
        self.bg_sub = BackgroundSubtractor() if use_bg_sub else None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.bg_sub is not None:
            frame = self.bg_sub.apply(frame)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        return frame

    def extract_sequences(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            return []

        frames: List[np.ndarray] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                frames.append(self._preprocess_frame(frame))
            frame_idx += 1

        cap.release()

        sequences: List[np.ndarray] = []
        for start in range(0, len(frames) - self.sequence_length + 1, self.sequence_length):
            sequences.append(np.stack(frames[start : start + self.sequence_length], axis=0))

        logger.info("%s: %s frames → %s sequences", Path(video_path).name, len(frames), len(sequences))
        return sequences

    def get_video_fps(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    def get_video_metadata(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {
            "fps": fps,
            "frames": frame_count,
            "width": width,
            "height": height,
            "duration_s": frame_count / fps if fps else 0.0,
        }


class VideoDatasetBuilder:
    """Scan videos, resolve labels, and build arrays only when labeling is valid."""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, raw_dir: str, processed_dir: str, config: dict):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.class_to_idx = config["dataset"]["class_to_idx"]
        self.extractor = FrameExtractor(
            height=config["preprocessing"]["frame_height"],
            width=config["preprocessing"]["frame_width"],
            sequence_length=config["preprocessing"]["sequence_length"],
            frame_skip=config["preprocessing"]["frame_skip"],
            use_bg_sub=config["preprocessing"].get("background_subtraction", True),
            normalize=config["preprocessing"].get("normalize", True),
        )
        self.report_dir = Path(config["paths"].get("report_dir", "outputs/reports"))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.label_manifest_path = Path(
            config["paths"].get("label_manifest", self.raw_dir / "labels.csv")
        )

    def _load_label_manifest(self) -> Dict[str, str]:
        if not self.label_manifest_path.exists():
            return {}

        manifest: Dict[str, str] = {}
        if self.label_manifest_path.suffix.lower() == ".json":
            with open(self.label_manifest_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                for key, value in payload.items():
                    manifest[str(key).lower()] = str(value).lower()
            return manifest

        with open(self.label_manifest_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = str(row.get("filename", "")).strip().lower()
                label = str(row.get("label", "")).strip().lower()
                if filename and label:
                    manifest[filename] = label
        return manifest

    @staticmethod
    def _normalise_label(label: str) -> Optional[str]:
        label = label.lower().strip()
        if label in {"low", "medium", "high"}:
            return label
        return None

    def _infer_label(self, filename: str, manifest: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str]:
        name_lower = filename.lower()
        if manifest:
            mapped = manifest.get(name_lower)
            if mapped:
                label = self._normalise_label(mapped)
                if label is not None:
                    return label, "manifest"

        for token in name_lower.replace("-", "_").split("_"):
            label = self._normalise_label(token)
            if label is not None:
                return label, "filename"

        return None, "unknown"

    def scan_videos(self) -> List[VideoRecord]:
        manifest = self._load_label_manifest()
        records: List[VideoRecord] = []

        for vf in sorted(self.raw_dir.iterdir()):
            if vf.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            cap = cv2.VideoCapture(str(vf))
            readable = cap.isOpened()
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if readable else 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if readable else 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if readable else 0
            cap.release()

            label, source = self._infer_label(vf.name, manifest)
            records.append(
                VideoRecord(
                    path=str(vf),
                    label=label,
                    label_source=source,
                    readable=readable,
                    fps=fps,
                    frame_count=frame_count,
                    width=width,
                    height=height,
                    duration_s=(frame_count / fps) if fps else 0.0,
                    size_mb=vf.stat().st_size / (1024 * 1024),
                )
            )

        return records

    def _write_scan_report(self, records: List[VideoRecord], reason: str) -> None:
        report = {
            "reason": reason,
            "raw_dir": str(self.raw_dir),
            "video_count": len(records),
            "labeled_count": sum(record.label is not None for record in records),
            "unlabeled_count": sum(record.label is None for record in records),
            "class_counts": {label: sum(record.label == label for record in records) for label in self.class_to_idx},
            "videos": [asdict(record) for record in records],
        }
        report_path = self.report_dir / "video_scan_report.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        csv_path = self.report_dir / "unlabeled_videos.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["filename", "label", "label_source", "readable", "fps", "frame_count", "width", "height", "duration_s", "size_mb"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "filename": Path(record.path).name,
                        "label": record.label or "",
                        "label_source": record.label_source,
                        "readable": record.readable,
                        "fps": record.fps,
                        "frame_count": record.frame_count,
                        "width": record.width,
                        "height": record.height,
                        "duration_s": record.duration_s,
                        "size_mb": round(record.size_mb, 3),
                    }
                )

    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        records = self.scan_videos()
        if not records:
            self._write_scan_report([], "no_supported_videos_found")
            raise FileNotFoundError(f"No supported video files found in {self.raw_dir}")

        all_sequences: List[np.ndarray] = []
        all_labels: List[int] = []
        sequence_manifest_rows: List[dict] = []
        skipped_records: List[dict] = []

        for record in records:
            video_path = Path(record.path)
            if record.label is None:
                skipped_records.append({"video": video_path.name, "reason": "unlabeled"})
                continue

            sequences = self.extractor.extract_sequences(str(video_path))
            if not sequences:
                skipped_records.append({"video": video_path.name, "reason": "too_short_or_unreadable"})
                continue

            label_idx = self.class_to_idx[record.label]
            for seq_idx in range(len(sequences)):
                sequence_manifest_rows.append(
                    {
                        "video_filename": video_path.name,
                        "video_path": str(video_path),
                        "sequence_index": seq_idx,
                        "label": record.label,
                        "label_idx": label_idx,
                    }
                )
            all_sequences.extend(sequences)
            all_labels.extend([label_idx] * len(sequences))

        if not all_sequences:
            self._write_scan_report(records, "no_labeled_sequences_extracted")
            raise ValueError(
                "No labeled sequences were extracted. The current raw dataset appears to be unlabeled or too short. "
                f"A scan report was written to {self.report_dir / 'video_scan_report.json'}"
            )

        X = np.stack(all_sequences, axis=0).astype(np.float32)
        y = np.asarray(all_labels, dtype=np.int32)

        np.save(self.processed_dir / "X.npy", X)
        np.save(self.processed_dir / "y.npy", y)

        manifest_path = self.processed_dir / "sequence_manifest.csv"
        with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["video_filename", "video_path", "sequence_index", "label", "label_idx"],
            )
            writer.writeheader()
            writer.writerows(sequence_manifest_rows)

        summary = {
            "raw_dir": str(self.raw_dir),
            "video_count": len(records),
            "sequence_count": int(len(X)),
            "class_counts": {label: int((y == idx).sum()) for label, idx in self.class_to_idx.items()},
            "skipped_records": skipped_records,
        }
        with open(self.report_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        self._write_scan_report(records, "processed_labeled_videos")
        logger.info("Saved dataset: X%s, y%s → %s", X.shape, y.shape, self.processed_dir)
        return X, y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess CCTV videos")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video_dir", default=None, help="Override raw video directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = args.video_dir or cfg["paths"]["raw_video_dir"]
    proc_dir = cfg["paths"]["processed_dir"]

    builder = VideoDatasetBuilder(raw_dir, proc_dir, cfg)
    builder.build()

"""
preprocessing.py  —  CCTV Video Preprocessing Pipeline
"""

import os
import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
import csv
import json
from tqdm import tqdm
from typing import List, Tuple, Optional

from typing import Dict
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
@dataclass
class VideoRecord:
    path: str
    label: Optional[str]
    label_source: str
    readable: bool
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    size_mb: float
    skipped_reason: Optional[str] = None


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BackgroundSubtractor:
    def __init__(self, history=200, var_threshold=50.0, detect_shadows=False):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

    def apply(self, frame):
        fg_mask = self.subtractor.apply(frame)
        fg_mask_3ch = cv2.merge([fg_mask, fg_mask, fg_mask])
        return cv2.bitwise_and(frame, fg_mask_3ch)


class FrameExtractor:
    def __init__(self, height=112, width=112, sequence_length=8,
                 frame_skip=4, use_bg_sub=True, normalize=True):
        self.height = height
        self.width = width
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.normalize = normalize
        self.bg_sub = BackgroundSubtractor() if use_bg_sub else None

    def _preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.bg_sub is not None:
            frame = self.bg_sub.apply(frame)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        return frame

    def extract_sequences(self, video_path):
        video_path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return []

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                frames.append(self._preprocess_frame(frame))
            frame_idx += 1
        cap.release()

        if not frames:
            logger.warning(f"No frames read from: {video_path}")
            return []

        sequences = []
        for start in range(0, len(frames) - self.sequence_length + 1,
                           self.sequence_length):
            seq = np.stack(frames[start: start + self.sequence_length], axis=0)
            sequences.append(seq)

        logger.info(f"  {Path(video_path).name}: {len(frames)} frames "
                    f"-> {len(sequences)} sequences")
        return sequences

    def get_video_fps(self, video_path):
        video_path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return max(fps, 1.0)

    def get_video_metadata(self, video_path):
        video_path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(video_path)
        fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0)
        meta = {
            "fps": fps,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps,
        }
        cap.release()
        return meta


class VideoDatasetBuilder:
    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, raw_dir, processed_dir, config):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.class_to_idx = config["dataset"]["class_to_idx"]
        self.extractor = FrameExtractor(
            height=config["preprocessing"]["frame_height"],
            width=config["preprocessing"]["frame_width"],
            sequence_length=config["preprocessing"]["sequence_length"],
            frame_skip=config["preprocessing"]["frame_skip"],
            use_bg_sub=config["preprocessing"]["background_subtraction"],
            normalize=config["preprocessing"]["normalize"],
        )

    def _infer_label(self, filename):
        name_lower = filename.lower()
        for label in self.class_to_idx:
            if label in name_lower:
                return label
        return None

    def build(self):
        video_files = [
            f for f in self.raw_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not video_files:
            logger.warning(f"No video files found in {self.raw_dir}.")
            return self._generate_demo_data()

        all_sequences = []
        all_labels = []

        for vf in tqdm(video_files, desc="Processing videos"):
            label = self._infer_label(vf.stem)
                    self.report_dir = Path(config["paths"].get("report_dir", "outputs/reports"))
                    self.report_dir.mkdir(parents=True, exist_ok=True)
                    self.label_manifest_path = Path(config["paths"].get("label_manifest", self.raw_dir / "labels.csv"))
            if label is None:
                def _load_label_manifest(self) -> Dict[str, str]:
                    """Load optional filename -> label mapping from CSV/JSON manifest."""
                    if not self.label_manifest_path.exists():
                        return {}

                    manifest: Dict[str, str] = {}
                    if self.label_manifest_path.suffix.lower() == ".json":
                        with open(self.label_manifest_path, "r", encoding="utf-8") as handle:
                            data = json.load(handle)
                        if isinstance(data, dict):
                            for key, value in data.items():
                                manifest[str(key).lower()] = str(value).lower()
                        return manifest

                    with open(self.label_manifest_path, "r", encoding="utf-8", newline="") as handle:
                        reader = csv.DictReader(handle)
                        for row in reader:
                            filename = str(row.get("filename", "")).strip().lower()
                            label = str(row.get("label", "")).strip().lower()
                            if filename and label:
                                manifest[filename] = label
                    return manifest

                @staticmethod
                def _normalise_label(label: str) -> Optional[str]:
                    label = label.lower().strip()
                    if label in {"low", "medium", "high"}:
                        return label
                    return None
                logger.warning(f"Skipping {vf.name}: no label in filename.")
                def _infer_label(self, filename: str, manifest: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str]:
            sequences = self.extractor.extract_sequences(str(vf))
            if not sequences:
                continue
            label_idx = self.class_to_idx[label]
            all_sequences.extend(sequences)
                    Priority order:
                      1. External label manifest (filename -> label)
                      2. Exact filename token matching (e.g. *_low_*.mp4)

                    Returns (label, source) where source is one of:
                      - manifest
                      - filename
                      - unknown
            all_labels.extend([label_idx] * len(sequences))

                    if manifest:
                        mapped = manifest.get(name_lower)
                        if mapped:
                            label = self._normalise_label(mapped)
                            if label is not None:
                                return label, "manifest"

                    tokens = [token for token in name_lower.replace("-", "_").split("_") if token]
                    for token in tokens:
                        label = self._normalise_label(token)
                        if label is not None:
                            return label, "filename"

                    return None, "unknown"

                def scan_videos(self) -> List[VideoRecord]:
                    """Scan raw_dir and return metadata for every supported video file."""
                    manifest = self._load_label_manifest()
                    records: List[VideoRecord] = []

                    for vf in sorted(self.raw_dir.iterdir()):
                        if vf.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                            continue

                        cap = cv2.VideoCapture(str(vf))
                        opened = cap.isOpened()
                        fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
                        cap.release()

                        label, source = self._infer_label(vf.name, manifest)
                        size_mb = vf.stat().st_size / (1024 * 1024)
                        records.append(VideoRecord(
                            path=str(vf),
                            label=label,
                            label_source=source,
                            readable=opened,
                            fps=fps,
                            frame_count=frame_count,
                            width=width,
                            height=height,
                            duration_s=(frame_count / fps) if fps else 0.0,
                            size_mb=size_mb,
                        ))

                    return records

                def _write_scan_report(self, records: List[VideoRecord], reason: str) -> None:
                    """Persist a machine-readable scan report for debugging and labeling."""
                    report = {
                        "reason": reason,
                        "raw_dir": str(self.raw_dir),
                        "video_count": len(records),
                        "labeled_count": sum(1 for record in records if record.label is not None),
                        "unlabeled_count": sum(1 for record in records if record.label is None),
                        "class_counts": {
                            label: sum(1 for record in records if record.label == label)
                            for label in self.class_to_idx
                        },
                        "videos": [asdict(record) for record in records],
                    }
                    report_path = self.report_dir / "video_scan_report.json"
                    with open(report_path, "w", encoding="utf-8") as handle:
                        json.dump(report, handle, indent=2)

                    manifest_path = self.report_dir / "unlabeled_videos.csv"
                    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["filename", "label", "label_source", "readable", "fps", "frame_count", "width", "height", "duration_s", "size_mb"])
                        writer.writeheader()
                        for record in records:
                            writer.writerow({
                                "filename": Path(record.path).name,
                                "label": record.label or "",
                                "label_source": record.label_source,
                                "readable": record.readable,
                                "fps": record.fps,
                                "frame_count": record.frame_count,
                                "width": record.width,
                                "height": record.height,
                                "duration_s": record.duration_s,
                                "size_mb": round(record.size_mb, 3),
                            })
        chunk_size = 100
                    video_files = self.scan_videos()

        logger.info(f"Writing {n_total} sequences in chunks of {chunk_size}...")

        X_mmap = np.lib.format.open_memmap(
                            "Please add supported videos or configure a label manifest."
        )
                        self._write_scan_report([], "no_supported_videos_found")
                        raise FileNotFoundError(
                            f"No supported video files found in {self.raw_dir}."
                        )
            chunk_seqs = all_sequences[i: i + chunk_size]
            chunk_arr = np.stack(chunk_seqs, axis=0)
            end = i + len(chunk_seqs)
                    skipped: List[dict] = []
            X_mmap[i:end] = chunk_arr
                    for record in tqdm(video_files, desc="Processing videos"):
                        vf = Path(record.path)
                        label = record.label
        del X_mmap
                            skipped.append({"video": vf.name, "reason": "unlabeled"})
        y = np.array(all_labels, dtype=np.int32)
        np.save(str(y_path), y)

        X = np.load(str(X_path), mmap_mode="r")
        logger.info(f"Dataset ready: X{X.shape}, y{y.shape}")

        for label, idx in self.class_to_idx.items():

                        if not sequences:
                            skipped.append({"video": vf.name, "reason": "too_short_or_unreadable"})
            count = int((y == idx).sum())
            logger.info(f"  {label:>8}: {count} sequences")
                        self._write_scan_report(video_files, "no_labeled_sequences_extracted")
                        raise ValueError(
                            "No labeled sequences were extracted. The dataset appears to be unlabeled or too short. "
                            f"A scan report was written to {self.report_dir / 'video_scan_report.json'}"
                        )

    def _generate_demo_data(self):
        logger.info("Generating synthetic demo dataset.")
        cfg = self.cfg["preprocessing"]
        T = cfg["sequence_length"]
        H = cfg["frame_height"]
        W = cfg["frame_width"]
        rng = np.random.default_rng(42)
        samples_per_class = 20
        sequences, labels = [], []

        for cls_idx, (_, brightness) in enumerate(
                [("low", 0.2), ("medium", 0.5), ("high", 0.8)]):
            for _ in range(samples_per_class):
                base = rng.normal(brightness, 0.05,
                                  (T, H, W, 3)).clip(0, 1).astype(np.float32)
                for t in range(T):
                    num_vehicles = int(cls_idx * 8 + rng.integers(0, 6))
                    for _ in range(num_vehicles):
                        x1 = int(rng.integers(0, max(W - 30, 1)))
                        y1 = int(rng.integers(0, max(H - 15, 1)))
                        x2 = min(x1 + 30, W)
                        y2 = min(y1 + 15, H)
                        base[t, y1:y2, x1:x2] = rng.uniform(
                            0.6, 1.0, (y2 - y1, x2 - x1, 3)
                        ).astype(np.float32)
                sequences.append(base)
                labels.append(cls_idx)

        X = np.stack(sequences, axis=0)
        y = np.array(labels, dtype=np.int32)
        np.save(self.processed_dir / "X.npy", X)
        np.save(self.processed_dir / "y.npy", y)
        logger.info(f"Demo dataset: X{X.shape}, y{y.shape}")
        return X, y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video_dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_dir = args.video_dir or cfg["paths"]["raw_video_dir"]
    builder = VideoDatasetBuilder(raw_dir, cfg["paths"]["processed_dir"], cfg)
    X, y = builder.build()
    logger.info({k: int((y == v).sum()) for k, v in cfg["dataset"]["class_to_idx"].items()})