from torchvision import transforms


def _convert_to_rgb(image):
    return image.convert('RGB')

def get_preprocess(image_resolution=224):

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )

    preprocess = transforms.Compose([
        transforms.Resize(
            size=image_resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_resolution),
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize,
    ])
    return preprocess