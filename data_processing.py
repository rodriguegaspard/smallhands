import os
import json
from pathlib import Path
import shutil
import random

from ultralytics.data.utils import compress_one_image
from ultralytics.utils.downloads import zip_directory

class_ids = [
        "dislike",
        "fist",
        "like",
        "one",
        "palm",
        "stop",
        "no_gesture"
        ]

class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Path to images and annotations
path_annot = "./datasets/annotations/"

json_files = sorted(
        [
            os.path.join(path_annot, filename)
            for filename in os.listdir(path_annot)
            ]
        )

def convert_xywh_to_cxcywh(bboxes):
    converted_bboxes = []
    for bbox in bboxes:
        converted_bbox = []
        converted_bbox.append(bbox[0] + (bbox[2]/2))
        converted_bbox.append(bbox[1] + (bbox[3]/2))
        converted_bbox.append(bbox[2])
        converted_bbox.append(bbox[3])
        converted_bboxes.append(converted_bbox)
    return converted_bboxes

def parse_annotations(json_file):
    images = []
    boxes = []
    labels = []

    with open(json_file) as f:
        print("Parsing " + json_file + "..")
        data = json.load(f)
        for filename in data:
            images.append(filename)
            boxes.append(convert_xywh_to_cxcywh(data[filename]['bboxes']))
            labels_ids = [list(class_mapping.keys())[list(class_mapping.values()).index(lbl)] for lbl in data[filename]['labels']]
            labels.append(labels_ids)
    return images, boxes, labels

def dataset_split(dataset, split_ratio):
    random.shuffle(dataset)
    index = int(len(dataset)*split_ratio)
    return dataset[:index], dataset[index:]

def create_dataset(json_files):
    dataset = []
    imgs = []
    bboxes = []
    clss = []
    # Parsing every file
    for file in json_files:
        images, boxes, labels = parse_annotations(file)
        imgs += images
        bboxes += boxes
        clss += labels
        dataset = [[imgs[i], bboxes[i], clss[i]] for i in range(len(imgs))]

    train,val = dataset_split(dataset, 0.8)

    os.makedirs('./datasets/hands/train/images', exist_ok=True)
    os.makedirs('./datasets/hands/train/labels', exist_ok=True)
    os.makedirs('./datasets/hands/val/images', exist_ok=True)
    os.makedirs('./datasets/hands/val/labels', exist_ok=True)

    # Writing the text files
    print("Writing labels and moving files..")
    for split in [("train", train), ("val", val)]:
        for data in split[1]:
            shutil.move("./datasets/images/" + data[0] + ".jpg", "./datasets/hands/" + split[0] + "/images/" + data[0] + ".jpg")
            with open("./datasets/hands/" + split[0] + "/labels/" + data[0] + ".txt", 'w') as file:
                for bbox, label in zip(data[1], data[2]):
                    # I might need to map those to a different number and a new class?
                    file.write(str(label) + " " + str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + "\n")

def optimize_dataset(directory):
    # Define dataset directory
    path = Path(directory)
    # Optimize images in dataset (optional)
    for f in path.rglob("*.jpg"):
        compress_one_image(f)
    # Zip dataset into 'path/to/dataset.zip'
    zip_directory(path)

create_dataset(json_files)
optimize_dataset("./datasets/hands/")

