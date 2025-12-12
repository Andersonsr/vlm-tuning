import json
import os
from tqdm import tqdm


if __name__ == '__main__':
    path = 'D:\\datasets\\nwpu\\nwpu_captions.json'
    data = json.load(open(path, 'r', encoding='utf-8'))

    output = {'train': [], 'val': [], 'test': []}

    for classification in data:
        for e in tqdm(data[classification]):
            file = os.path.join(classification, e['filename'])
            texts = []
            for i in range(5):
                if i == 0:
                    texts.append(e['raw'])
                else:
                    texts.append(e[f'raw_{i}'])

            sample = {
                'image_name': file,
                'image_id': e['imgid'],
                'captions': texts,
                'class': classification,
            }
            output[e['split']].append(sample)

    for key in output.keys():
        json.dump(output[key], open(f'D:\\datasets\\nwpu\\{key}.json', 'w'), indent=2)

