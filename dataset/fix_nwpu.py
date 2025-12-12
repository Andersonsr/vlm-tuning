import os
from glob import glob

if __name__ == '__main__':
    dest_dataset = 'D:\\datasets\\nwpu\\images'

    for split in ['D:\\datasets\\archive\\Dataset\\train\\train\\*', 'D:\\datasets\\archive\\Dataset\\test\\test\\*']:
        origin_classes = glob(split)
        for class_path in origin_classes:
            class_name = os.path.basename(class_path)
            images = glob(os.path.join(class_path, '*'))
            existing_images = glob(os.path.join(dest_dataset, class_name, '*'))
            existing_images = [os.path.basename(e) for e in existing_images]
            # print(existing_images)
            for image in images:
                image_name = os.path.basename(image)
                if image_name not in existing_images:
                    print(image_name)