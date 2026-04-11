import os, shutil, warnings
from xml.etree import ElementTree as ET
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
train_dir  = os.path.join(BASE_DIR, "train_data", "train")
test_dir   = os.path.join(BASE_DIR, "test_data",  "test")
dest_dataset = os.path.join(BASE_DIR, "dataset")
images_dir   = os.path.join(dest_dataset, "images")
labels_dir   = os.path.join(dest_dataset, "labels")

class_mapping = {"apple": 0, "banana": 1, "orange": 2}

# ── Helpers ──────────────────────────────────────────────
def convert_pvoc_to_yolo(width, height, bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center   = (x_min + x_max) / 2 / (width  + 1e-6)
    y_center   = (y_min + y_max) / 2 / (height + 1e-6)
    box_width  = (x_max - x_min)     / (width  + 1e-6)
    box_height = (y_max - y_min)     / (height + 1e-6)
    return x_center, y_center, box_width, box_height

def parse_xml_to_yolo(src_dir, split):
    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
    print(f"  Found {len(image_files)} images in {split}")
    copied = 0
    for image_file in image_files:
        src_img   = os.path.join(src_dir, image_file)
        src_label = os.path.join(src_dir, os.path.splitext(image_file)[0] + ".xml")
        dst_img   = os.path.join(images_dir, split, image_file)
        dst_label = os.path.join(labels_dir, split, os.path.splitext(image_file)[0] + ".txt")

        if not os.path.exists(src_label):
            continue

        shutil.copy2(src_img, dst_img)

        try:
            tree   = ET.parse(src_label)
            root   = tree.getroot()
            size   = root.find('size')
            width  = int(size.find('width').text)
            height = int(size.find('height').text)

            with open(dst_label, 'w') as f:
                for obj in root.findall('object'):
                    name_tag = obj.find('name')
                    if name_tag is None:
                        continue
                    class_label = name_tag.text.strip().lower()
                    if class_label not in class_mapping:
                        continue
                    class_id  = class_mapping[class_label]
                    bbox_node = obj.find('bndbox')
                    if bbox_node is None:
                        continue
                    bbox = (
                        float(bbox_node.find('xmin').text),
                        float(bbox_node.find('ymin').text),
                        float(bbox_node.find('xmax').text),
                        float(bbox_node.find('ymax').text),
                    )
                    yolo_box = convert_pvoc_to_yolo(width, height, bbox)
                    f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")
            copied += 1
        except Exception as e:
            print(f"  Error parsing {image_file}: {e}")

    print(f"  Copied: {copied}")

# ────────────────────────────────────────────────────────
# ✅ THIS IS THE FIX — required on Windows for multiprocessing
# ────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Wipe old dataset
    if os.path.exists(dest_dataset):
        shutil.rmtree(dest_dataset)
        print("Cleared old dataset folder.")

    # Create fresh dirs
    for split in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    # Convert dataset
    print("\n=== Converting dataset ===")
    parse_xml_to_yolo(train_dir, 'train')
    parse_xml_to_yolo(test_dir,  'val')
    print("Conversion done!\n")

    # Write data.yaml with absolute paths
    train_img_dir = os.path.join(images_dir, 'train').replace(os.sep, '/')
    val_img_dir   = os.path.join(images_dir, 'val').replace(os.sep, '/')

    yaml_content = f"""train: {train_img_dir}
val: {val_img_dir}

nc: 3
names: ['apple', 'banana', 'orange']
"""
    yaml_path = os.path.join(BASE_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"data.yaml written.\n")

    # Train
    print("=== Starting Training ===")
    model = YOLO('yolov8n.yaml')
    model.train(
        data    = yaml_path,
        epochs  = 100,
        batch   = 16,
        imgsz   = 640,
        workers = 2,        # ← Windows needs this inside __main__
        device  = 0
    )

    print("\nDone! Weights at: runs/detect/train/weights/best.pt")