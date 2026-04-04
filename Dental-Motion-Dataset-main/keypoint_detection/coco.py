import os
import json
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TXT keypoints to COCO format JSON")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to image and txt files (e.g., data/teeth/train/images)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON path (e.g., data/teeth/annotations/person_keypoints_train.json)')
    return parser.parse_args()


def load_keypoints(txt_file):
    keypoints = []

    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            try:
                parts = line.split(",")
                if len(parts) == 2:
                    x, y = map(float, parts)
                    keypoints.extend([x, y, 2])
            except ValueError:
                print(f"[Warning] Invalid line in {txt_file}: {line}")

    return keypoints


def get_image_size(img_path):
    try:
        import cv2
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
    except:
        h, w = 1080, 1920
    return h, w


def main():
    args = parse_args()

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "keypoints": [],
                "skeleton": []
            }
        ]
    }

    ann_id = 1
    img_id = 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for fname in tqdm(os.listdir(args.data_root)):
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        base = os.path.splitext(fname)[0]
        img_path = os.path.join(args.data_root, fname)
        txt_path = os.path.join(args.data_root, base + ".txt")

        if not os.path.exists(txt_path):
            continue

        keypoints = load_keypoints(txt_path)

        if not keypoints:
            continue

        # 初始化 keypoint names
        if not coco_dict["categories"][0]["keypoints"]:
            num_kpt = len(keypoints) // 3
            coco_dict["categories"][0]["keypoints"] = [f"kpt_{i}" for i in range(num_kpt)]

        x_coords = keypoints[0::3]
        y_coords = keypoints[1::3]

        bbox = [
            min(x_coords),
            min(y_coords),
            max(x_coords) - min(x_coords),
            max(y_coords) - min(y_coords)
        ]

        height, width = get_image_size(img_path)

        coco_dict["images"].append({
            "id": img_id,
            "file_name": fname,
            "height": height,
            "width": width
        })

        coco_dict["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": len(keypoints) // 3,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })

        ann_id += 1
        img_id += 1

    with open(args.output, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"✅ Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
