import os
import shutil
from collections import defaultdict
from PIL import Image

def normalize_class_name(name):
    # Strip parenthetical subtypes
    return name.split('(')[0].strip()

def load_bounding_boxes():
    bounding_boxes = {}
    with open('nabirds/bounding_boxes.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                img_id, x, y, w, h = parts
                bounding_boxes[img_id] = (float(x), float(y), float(w), float(h))
    return bounding_boxes

def create_dataset():
    # Create base directories
    os.makedirs('nabirds-dataset/train', exist_ok=True)
    os.makedirs('nabirds-dataset/test', exist_ok=True)

    # Read class names and create normalized mapping
    class_dict = {}
    normalized_classes = defaultdict(set)
    with open('nabirds/classes.txt', 'r') as f:
        for line in f:
            class_id, name = line.strip().split(' ', 1)
            normalized_name = normalize_class_name(name)
            class_dict[class_id] = normalized_name
            normalized_classes[normalized_name].add(class_id)

    # Create reverse mapping from normalized names to a representative class ID
    normalized_to_id = {name: min(class_ids) for name, class_ids in normalized_classes.items()}

    # Read train/test split
    is_train = {}
    with open('nabirds/train_test_split.txt', 'r') as f:
        for line in f:
            img_id, is_training = line.strip().split()
            is_train[img_id] = is_training == "1"

    # Read image paths
    image_paths = {}
    with open('nabirds/images.txt', 'r') as f:
        for line in f:
            img_id, path = line.strip().split()
            image_paths[img_id] = path

    # Read image labels
    image_labels = {}
    with open('nabirds/image_class_labels.txt', 'r') as f:
        for line in f:
            img_id, class_id = line.strip().split()
            normalized_name = class_dict[class_id]
            # Use the representative class ID for the normalized name
            image_labels[img_id] = normalized_to_id[normalized_name]

    # Load bounding boxes
    bounding_boxes = load_bounding_boxes()

    # Tracker for test images count per class
    test_tracker = defaultdict(int)

    # Copy and process images to appropriate directories
    num_train = 0
    num_test = 0
    for img_id, path in image_paths.items():
        if img_id not in image_labels:
            continue

        class_id = image_labels[img_id]
        class_name = class_dict[class_id]
        
        # Determine initial split from train_test_split file
        original_split = 'train' if is_train[img_id] else 'test'
        if original_split == 'test':
            if test_tracker[class_name] < 15:
                split = 'test'
                test_tracker[class_name] += 1
            else:
                split = 'train'
        else:
            split = 'train'
        
        src = os.path.join('nabirds', 'images', path)
        dst = os.path.join('nabirds-dataset', split, class_name, os.path.basename(path))
        
        with Image.open(src) as im:
            x, y, w, h = bounding_boxes[img_id]
            center_x = x + w / 2
            center_y = y + h / 2
            side = max(w, h) * 1.2
            left = center_x - side / 2
            top = center_y - side / 2
            right = center_x + side / 2
            bottom = center_y + side / 2

            # Clamp crop coordinates to image boundaries
            left = max(left, 0)
            top = max(top, 0)
            right = min(right, im.width)
            bottom = min(bottom, im.height)

            im_cropped = im.crop((left, top, right, bottom))

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if im_cropped.mode != 'RGB':
                im_cropped = im_cropped.convert('RGB')
            im_cropped.save(dst)
            if split == 'train':
                num_train += 1
            else:
                num_test += 1

            print(f"Processed {num_train + num_test}")
    return {
        'num_classes': len(normalized_classes),
        'num_train': num_train,
        'num_test': num_test
    }

def main():
    print("Creating PyTorch-compatible dataset...")
    stats = create_dataset()
    
    print("\nDataset creation complete!")
    print(f"Number of normalized classes: {stats['num_classes']}")
    print(f"Training images: {stats['num_train']}")
    print(f"Testing images: {stats['num_test']}")
    print(f"Total images: {stats['num_train'] + stats['num_test']}")

if __name__ == '__main__':
    main()
