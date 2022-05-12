import json
import os
from PIL import Image
import numpy as np
import datetime
from pycococreatortools import pycococreatortools
from tqdm import tqdm
p = 'Dataset'


INFO = {
    "description": "Pothole Segmentation",
    "url": "https://github.com/andrijdavid/Cracks-and-Porholes-in-Road-Images-Dataset",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "andrijdavid",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'crack',
        'supercategory': 'crack',
    },
    {
        'id': 2,
        'name': 'lane',
        'supercategory': 'lane',
    },
    {
        'id': 3,
        'name': 'pothole',
        'supercategory': 'pothole',
    },
]

coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }


dirs = os.listdir(p)
image_id = 1
segmentation_id = 1

for d in tqdm(dirs):
    image_filename = f"{p}/{d}/{d}_RAW.jpg"
    image = Image.open(image_filename)
    image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
    coco_output["images"].append(image_info)
    for a in ["CRACK", "LANE", "POTHOLE"]:
        annotation_filename = f"{p}/{d}/{d}_{a}.png"
        class_id = [x['id'] for x in CATEGORIES if x['name'].lower() == a.lower() ][0]

        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
        binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)
        annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)
        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)
            segmentation_id = segmentation_id + 1
        image_id = image_id + 1

with open('coco.json', 'w') as output_json_file:
     json.dump(coco_output, output_json_file)
