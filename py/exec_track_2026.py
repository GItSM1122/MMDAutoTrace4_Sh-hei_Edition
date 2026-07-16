"""Colab 2026 compatibility entry for MMDAutoTrace4.

This module deliberately does not use the legacy PHALP tracker path.  The
first goal is to validate the installed 4D-Humans/HMR2 environment and model
availability without triggering PHALP's obsolete Berkeley cache downloads.

Once this probe succeeds, the actual per-frame inference and MMDAutoTrace4 PKL
adapter can be implemented here while keeping py/exec_track.py unchanged.
"""

import os


def run(video_path: str, output_dir: str) -> None:
    print("Start: 4D-Humans 2026 compatibility check =============================")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    os.makedirs(output_dir, exist_ok=True)

    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Import HMR2 directly.  Do not import PHALP here: legacy PHALP startup
    # attempts to download cache assets from URLs that now return 403/404.
    from hmr2.models import download_models, load_hmr2

    print("Downloading/checking 4D-Humans model files ...")
    download_models()

    print("Loading HMR2 model ...")
    model, model_cfg = load_hmr2()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    marker = os.path.join(output_dir, "hmr2_2026_ready")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("HMR2 model loaded successfully.\n")

    print("HMR2 model loaded successfully.")
    print("Ready marker:", marker)
    print("Next step: implement video frame inference and PKL compatibility adapter.")
    print("End: 4D-Humans 2026 compatibility check ===============================")
