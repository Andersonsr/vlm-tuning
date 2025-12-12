import json

from pycocotools.coco import COCO
from tqdm import tqdm
import os

if __name__ == '__main__':
    splits = ['train', 'val', ]
    datasetRoot = 'D:\\datasets\\coco_2017'

    for split in splits:
        data = []
        savePath = f'D:\\datasets\\coco_2017\\{split}.json'
        dirname = os.path.dirname(savePath)
        os.makedirs(dirname, exist_ok=True)

        path = os.path.join(datasetRoot, 'annotations', f'captions_{split}2017.json')
        coco = COCO(path)
        ids = coco.getImgIds()
        imgs = coco.loadImgs(ids)

        for i, image in enumerate(tqdm(imgs)):
            ann = coco.loadAnns(coco.getAnnIds(ids[i]))
            texts = [e['caption'] for e in ann]
            sample = {
                'image_name': image['file_name'],
                'image_id': ids[i],
                'captions': texts[:5]
            }
            data.append(sample)

        with open(savePath, 'w') as f:
            json.dump(data, f, indent=2)
