#!/usr/bin/env python3
"""Build deterministic test-only corruptions for the paired OGSOD-1.0 test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


SEED = 3407
DATA_ROOT = Path("/opt/data/private/DATA/OGSOD-1.0")
OUTPUT_ROOT = DATA_ROOT / "robustness_test"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
CONDITIONS = ("optical_degradation", "modal_misalignment", "sar_noise")
SEVERITIES = ("mild", "moderate", "severe")
OPTICAL = {
    "mild": (3, (0.5, 1.0), (0.80, 0.90)),
    "moderate": (5, (1.0, 2.0), (0.60, 0.75)),
    "severe": (9, (2.0, 3.0), (0.40, 0.60)),
}
SHIFTS = {"mild": 4, "moderate": 8, "severe": 16}
NOISE_SIGMAS = {"mild": 0.05, "moderate": 0.10, "severe": 0.20}
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--severity", action="append", choices=SEVERITIES)
    parser.add_argument("--limit", type=int, default=None, help="Build only the first N pairs (smoke tests only).")
    parser.add_argument("--copy-links", action="store_true", help="Copy unchanged files instead of symlinking them.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing condition directories.")
    return parser.parse_args()


def image_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    result = {p.stem: p.resolve() for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if not result:
        raise RuntimeError(f"No images found in {directory}")
    return result


def discover_pairs(data_root: Path, limit: int | None = None) -> list[dict[str, Path | str]]:
    rgb = image_map(data_root / "rgb/images/test")
    sar = image_map(data_root / "sar/images/test")
    if rgb.keys() != sar.keys():
        raise RuntimeError(f"RGB/SAR pairing mismatch: RGB-only={sorted(rgb.keys()-sar.keys())[:5]}, SAR-only={sorted(sar.keys()-rgb.keys())[:5]}")
    pairs = []
    for sample_id in sorted(rgb):
        label = data_root / "rgb/labels/test" / f"{sample_id}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {sample_id}: {label}")
        pairs.append({"sample_id": sample_id, "rgb": rgb[sample_id], "sar": sar[sample_id], "label": label.resolve()})
    return pairs[:limit] if limit is not None else pairs


def sample_seed(seed: int, condition: str, severity: str, sample_id: str) -> int:
    payload = f"{seed}|{condition}|{severity}|{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def dtype_limits(image: np.ndarray) -> tuple[float, float]:
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(image.dtype, np.floating):
        finite = image[np.isfinite(image)]
        if not finite.size:
            raise ValueError("Floating-point image contains no finite pixels")
        return float(finite.min()), float(finite.max())
    raise TypeError(f"Unsupported image dtype: {image.dtype}")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV failed to read {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV failed to write {path}")


def apply_optical_degradation(image: np.ndarray, kernel: int, sigma: float, brightness: float) -> np.ndarray:
    lo, hi = dtype_limits(image)
    output = cv2.GaussianBlur(image.astype(np.float32), (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    output = np.clip(output * brightness, lo, hi)
    if np.issubdtype(image.dtype, np.integer):
        output = np.rint(output)
    return output.astype(image.dtype)


def apply_translation(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def apply_speckle_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    lo, hi = dtype_limits(image)
    span = hi - lo
    if span <= 0:
        return image.copy()
    normalized = (image.astype(np.float32) - lo) / span
    noise = rng.normal(0.0, sigma, image.shape[:2]).astype(np.float32)
    if normalized.ndim == 3:
        noise = noise[..., None]  # shared spatial noise avoids false chromatic artifacts
    output = np.clip(normalized * (1.0 + noise), 0.0, 1.0) * span + lo
    if np.issubdtype(image.dtype, np.integer):
        output = np.rint(output)
    return output.astype(image.dtype)


def apply_modality_dropout(image: np.ndarray, fill: str = "zero") -> np.ndarray:
    """Reserved, non-default helper for future optical/SAR dropout experiments."""
    if fill == "zero":
        return np.zeros_like(image)
    if fill == "mean":
        return np.full_like(image, np.mean(image, axis=(0, 1), keepdims=True))
    raise ValueError("fill must be 'zero' or 'mean'")


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
    else:
        os.symlink(source, destination)


def base_record(pair: dict, rgb_out: Path, sar_out: Path, condition: str, severity: str, seed: int) -> dict:
    return {
        "sample_id": pair["sample_id"], "rgb_original_path": str(pair["rgb"]), "sar_original_path": str(pair["sar"]),
        "rgb_output_path": str(rgb_out.resolve()), "sar_output_path": str(sar_out.resolve()), "label_path": str(pair["label"]),
        "condition": condition, "severity": severity, "modified_modality": "rgb" if condition == "optical_degradation" else "sar",
        "is_rgb_modified": condition == "optical_degradation", "is_sar_modified": condition != "optical_degradation", "random_seed": seed,
    }


def write_manifests(records: list[dict], manifests: Path, condition: str, severity: str) -> None:
    manifests.mkdir(parents=True, exist_ok=True)
    stem = f"{condition}_{severity}"
    keys = list(dict.fromkeys(k for record in records for k in record))
    with (manifests / f"{stem}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader(); writer.writerows(records)
    with (manifests / f"{stem}.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


def write_dataset_yaml(path: Path, subset_root: Path, clean_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Auto-generated OGSOD-1.0 robustness evaluation config.\npath: {subset_root}\ntrain: {clean_root / 'rgb/images/train'}\nval: rgb/images/test\ntest: rgb/images/test\ntrain1: {clean_root / 'sar/images/train'}\nval1: {subset_root / 'sar/images/test'}\ntest1: {subset_root / 'sar/images/test'}\nnames:\n  0: bridge\n  1: harbor\n  2: oil_tank\n"""
    path.write_text(content, encoding="utf-8")


def verify_subset(pairs: list[dict], root: Path, condition: str) -> None:
    rgb_out, sar_out, labels = image_map(root / "rgb/images/test"), image_map(root / "sar/images/test"), root / "rgb/labels/test"
    expected = {str(p["sample_id"]) for p in pairs}
    if set(rgb_out) != expected or set(sar_out) != expected:
        raise RuntimeError(f"Pair/count mismatch in {root}")
    for pair in pairs:
        sid = str(pair["sample_id"])
        if not (labels / f"{sid}.txt").is_file(): raise FileNotFoundError(labels / f"{sid}.txt")
        rgb0, sar0, rgb1, sar1 = read_image(pair["rgb"]), read_image(pair["sar"]), read_image(rgb_out[sid]), read_image(sar_out[sid])
        if rgb0.shape != rgb1.shape or sar0.shape != sar1.shape: raise RuntimeError(f"Resolution changed for {sid}")
        if condition == "optical_degradation" and not np.array_equal(sar0, sar1): raise RuntimeError(f"SAR changed for {sid}")
        if condition != "optical_degradation" and not np.array_equal(rgb0, rgb1): raise RuntimeError(f"RGB changed for {sid}")


def build_subset(pairs: list[dict], output_root: Path, clean_root: Path, condition: str, severity: str, seed: int, copy: bool, overwrite: bool) -> None:
    root = output_root / condition / severity
    if root.exists() and overwrite: shutil.rmtree(root)
    if root.exists(): raise FileExistsError(f"Subset already exists (use --overwrite): {root}")
    records = []
    for pair in pairs:
        sid_seed = sample_seed(seed, condition, severity, str(pair["sample_id"]))
        rng = np.random.default_rng(sid_seed)
        rgb_out = root / "rgb/images/test" / Path(pair["rgb"]).name
        sar_out = root / "sar/images/test" / Path(pair["sar"]).name
        label_out = root / "rgb/labels/test" / Path(pair["label"]).name
        record = base_record(pair, rgb_out, sar_out, condition, severity, sid_seed)
        if condition == "optical_degradation":
            kernel, sigma_range, brightness_range = OPTICAL[severity]
            sigma, brightness = rng.uniform(*sigma_range), rng.uniform(*brightness_range)
            write_image(rgb_out, apply_optical_degradation(read_image(pair["rgb"]), kernel, sigma, brightness))
            link_or_copy(pair["sar"], sar_out, copy)
            record.update(gaussian_kernel=kernel, gaussian_sigma=float(sigma), brightness_factor=float(brightness))
        elif condition == "modal_misalignment":
            shift = SHIFTS[severity]; ux, uy = DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]; dx, dy = shift * ux, shift * uy
            link_or_copy(pair["rgb"], rgb_out, copy); write_image(sar_out, apply_translation(read_image(pair["sar"]), dx, dy))
            direction = f"({'+' if dx > 0 else ''}{dx},{'+' if dy > 0 else ''}{dy})"
            record.update(dx=dx, dy=dy, shift_pixels=shift, translation_direction=direction, border_mode="cv2.BORDER_REFLECT_101")
        else:
            sigma = NOISE_SIGMAS[severity]
            link_or_copy(pair["rgb"], rgb_out, copy); write_image(sar_out, apply_speckle_noise(read_image(pair["sar"]), sigma, rng))
            record.update(noise_type="multiplicative_gaussian_speckle", noise_sigma=sigma, noise_seed=sid_seed)
        link_or_copy(pair["label"], label_out, copy)
        records.append(record)
    verify_subset(pairs, root, condition)
    write_manifests(records, output_root / "manifests", condition, severity)
    write_dataset_yaml(output_root / "configs" / f"{condition}_{severity}.yaml", root, clean_root)
    logging.info("%s/%s: %d OK", condition, severity, len(records))


def main() -> None:
    args = parse_args(); logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.output_root.resolve() == args.data_root.resolve(): raise ValueError("Output root must not be the original dataset root")
    pairs = discover_pairs(args.data_root.resolve(), args.limit)
    logging.info("Original test pairs: %d%s", len(pairs), " (limited smoke test)" if args.limit else "")
    for condition in args.condition or CONDITIONS:
        for severity in args.severity or SEVERITIES:
            build_subset(pairs, args.output_root.resolve(), args.data_root.resolve(), condition, severity, args.seed, args.copy_links, args.overwrite)


if __name__ == "__main__": main()
