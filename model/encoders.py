import os
import torch
import clip
from PIL import Image
from torch import nn


class CLIP(nn.Module):
    def __init__(self, conf):
        super(CLIP, self).__init__()
        modelName = conf.name.split(':')[1]
        self.model, self.preprocess = clip.load(modelName, device='cpu')
        self.tokenize = clip.tokenize
        self.model.logit_scale = torch.nn.Parameter(torch.log(torch.ones(1) * conf.temperature), requires_grad=conf.train_temperature)
        # print('temperature', self.model.logit_scale)

    def prepareImages(self, images: list[str],) -> torch.Tensor:
        """
        :param images: list of paths to images
        :return: images embeddings
        """
        inputs = []
        for image in images:
            # print(image)
            input = self.preprocess(Image.open(image))
            inputs.append(input)

        return torch.stack(inputs)

    def encode_image(self, image):
        x = self.model.encode_image(image)
        if hasattr(self, 'vision_adapter'):
            x = self.vision_adapter(x)
        return x

    def encode_text(self, text):
        x = self.model.encode_text(text)
        if hasattr(self, 'text_adapter'):
            x = self.text_adapter(x)
        return x

    def set_temperature(self, temperature):
        new_temp = torch.nn.Parameter(torch.log(torch.ones(1) * temperature),requires_grad=self.model.logit_scale.requires_grad)
        self.model.logit_scale = new_temp
        self.model.logit_scale.to()

    def forward(self, image, text):
        ce = torch.nn.CrossEntropyLoss()
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)

        # adapters
        if hasattr(self, 'vision_adapter'):
            image_features = self.vision_adapter(image_features)

        if hasattr(self, 'text_adapter'):
            text_features = self.text_adapter(text_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale.to(image_features.device) * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=logits_per_image.device)
        return (ce(logits_per_image, ground_truth) + ce(logits_per_text, ground_truth)) / 2, logits_per_image

    def learnable_parameters(self):
        learnable = 0
        total = 0
        for param in self.model.parameters():
            total += param.numel()
            if param.requires_grad:
                learnable += param.numel()

        print(f'total params: {total / 1e6:.2f}M,  learnable params: {learnable / 1e6:.2f}M')
        return total, learnable


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = CLIP("ViT-B/32").to(device)

    images = ["D:\\datasets\\coco_2017\\val2017\\000000000285.jpg", "D:\\datasets\\coco_2017\\val2017\\000000001584.jpg"]
    texts = ["a photo of a brown bear", 'a photo of a red bus in the street']
    with torch.no_grad():
        vis_embeddings = model.forwardImages(images, device)
        print(vis_embeddings.shape)
        tex_embeddings = model.forwardTexts(texts, device)
        print(tex_embeddings.shape)
