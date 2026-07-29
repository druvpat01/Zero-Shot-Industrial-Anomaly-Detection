"""Copy out the three frames ``docs/demo_script.md`` uploads, plus a blurred one.

The demo needs a clean part, a defective part and a frame the input-quality guard
will refuse. The first two are in the dataset. The third is not — MVTec's images
are all in focus, because it is a dataset about defective *objects*, not about
defective *photographs* — so it is made here, by blurring one of them until its
Laplacian variance falls under ``BLUR_THRESHOLD``.

That the blur has to be synthetic is worth saying out loud during the demo. The
guard exists for a failure mode this dataset does not contain and a production
line has constantly: a smeared lens, a part moving through the exposure, a light
that died. The model would happily score any of them, confidently and wrongly.

Usage::

    python scripts/make_demo_frames.py
    python scripts/make_demo_frames.py --category bottle --blur-sigma 8

Writes to ``results/demo_frames/`` (gitignored, like everything derived from the
dataset). Deterministic: the same inputs give byte-identical outputs, so a demo
rehearsed on Monday shows the same three pictures on Friday.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "MVTecAD"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "demo_frames"

#: Matches ``BLUR_RESIZE_EDGE`` in ``app/guardrails/quality.py``: the guard
#: measures sharpness at a fixed scale, so a 900x900 frame and a 256x256 crop of
#: it are judged on the same ramp. Reproduced here only to *report* the number
#: this script is aiming under — the verdict still comes from the API.
GUARD_RESIZE_EDGE = 256


def laplacian_variance(image: np.ndarray, resize_edge: int = GUARD_RESIZE_EDGE) -> float:
    """Sharpness, the way the guard measures it: variance of the Laplacian at a fixed scale."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = grey.shape[:2]
    short_edge = min(height, width)
    if short_edge != resize_edge and short_edge > 0:
        scale = resize_edge / short_edge
        grey = cv2.resize(grey, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="bottle", help="MVTec category to pull frames from.")
    parser.add_argument("--defect-type", default="broken_large", help="Test subfolder for the defective frame.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="MVTec AD root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write the frames.")
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=8.0,
        help="Gaussian sigma for the guard-rejection frame. Higher is blurrier; 8 is well under the default threshold of 50.",
    )
    args = parser.parse_args(argv)

    test_dir = args.data_root / args.category / "test"
    good_dir, defect_dir = test_dir / "good", test_dir / args.defect_type
    for directory in (good_dir, defect_dir):
        if not directory.is_dir():
            print(f"error: {directory} not found. Run scripts/download_dataset.py first.", file=sys.stderr)
            return 1

    good_src = sorted(good_dir.glob("*.png"))[0]
    defect_src = sorted(defect_dir.glob("*.png"))[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Copied rather than re-encoded: the demo should upload the same bytes the
    # benchmark scored, not a frame this script has quietly recompressed.
    clean_dst = args.output_dir / f"1_clean_{args.category}.png"
    defect_dst = args.output_dir / f"2_defective_{args.category}_{args.defect_type}.png"
    shutil.copyfile(good_src, clean_dst)
    shutil.copyfile(defect_src, defect_dst)

    frame = cv2.imread(str(good_src), cv2.IMREAD_COLOR)
    # ksize=(0, 0) lets OpenCV derive the kernel from sigma, so --blur-sigma is
    # the only dial and it cannot be set to something the kernel contradicts.
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=args.blur_sigma, sigmaY=args.blur_sigma)
    blurred_dst = args.output_dir / f"3_blurred_{args.category}.png"
    cv2.imwrite(str(blurred_dst), blurred)

    sharp_variance = laplacian_variance(frame)
    blurred_variance = laplacian_variance(blurred)

    print(f"clean      {clean_dst.relative_to(REPO_ROOT)}  (laplacian variance {sharp_variance:.1f})")
    print(f"defective  {defect_dst.relative_to(REPO_ROOT)}")
    print(f"blurred    {blurred_dst.relative_to(REPO_ROOT)}  (laplacian variance {blurred_variance:.1f}, sigma {args.blur_sigma})")
    if blurred_variance >= 50.0:
        print(
            f"warning: {blurred_variance:.1f} is above the default BLUR_THRESHOLD of 50 — "
            "the guard will accept this frame. Raise --blur-sigma.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
