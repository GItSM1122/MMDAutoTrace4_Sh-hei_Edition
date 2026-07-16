from datetime import datetime
import os
import sys

import exec_track_2026


def main():
    if len(sys.argv) < 2:
        print("Usage: python py/exec_gpu_2026.py VIDEO_PATH [OUTPUT_DIR]")
        sys.exit(2)

    video_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if len(sys.argv) > 2:
        output_dir = os.path.abspath(sys.argv[2])
    else:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(
            os.path.dirname(video_path),
            f"{stem}_2026_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    os.makedirs(output_dir, exist_ok=True)
    print("Video:", video_path)
    print("Output:", output_dir)

    exec_track_2026.run(video_path, output_dir)


if __name__ == "__main__":
    main()
