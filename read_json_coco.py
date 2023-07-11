import cv2
import json
import numpy as np
from tqdm import tqdm

f = open("Smoke-detector-4\\train\\annot\\_annotations.coco.json", 'r')
data = json.load(f)
images = data['images']
annots = data['annotations']
for x in tqdm(images, total=len(images)):
    id_ = x['id']
    filename = x['file_name']
    h = x['height']
    w = x['width']
    for y in annots:
        mask = np.zeros((w, h), dtype=np.int32)
        cat = y['category_id']
        img_id = y['image_id']
        if not (cat == 1) and img_id == id_:
            seg = y['segmentation']
            for points in seg:
                contours = []
                for i in range(0, len(points), 2):
                    contours.append([points[i], points[i+1]])
                contours = np.array(contours, dtype=np.int32)
                cv2.fillPoly(img=mask, pts=[contours], color=(255, 0, 0))
            cv2.imwrite("Smoke-detector-4\\masks\\"+filename, mask)