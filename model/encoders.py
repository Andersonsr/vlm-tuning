import torch
import clip
from PIL import Image
import lightning as L
from torchmetrics.retrieval import RetrievalRecall
from torch.optim import AdamW, Adam
from lora_utils import mark_only_lora_as_trainable


class CLIP(L.LightningModule):
    def __init__(self, conf):
        super(CLIP, self).__init__()
        modelName = conf.model.name.split(':')[1]
        self.model, self.preprocess = clip.load(modelName, device='cpu')
        self.tokenize = clip.tokenize
        self.automatic_optimization = False

        self.model.logit_scale = torch.nn.Parameter(
            torch.log(torch.ones(1) * conf.model.temperature),
            requires_grad=conf.model.train_temperature
        )

        self.train_temperature = conf.model.train_temperature
        self.cooling = None
        self.lr = conf.train.learning_rate
        self.lora = conf.model.lora.apply

        if conf.train.cooling.apply:
            self.cooling = conf.train.cooling.apply
            self.target_temperature = conf.train.cooling.final_temp
            self.cooling_steps = conf.train.cooling.iterations
            self.step = 0

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

    def update_temperature(self):
        if self.cooling == 'linear':
            cooling_rate = (100 - self.target_temperature) / self.cooling_steps
            temperature = max(100.0 - (self.step * cooling_rate), self.target_temperature)

        elif self.cooling == 'step':
            delta = 100.0 - self.target_temperature
            num_steps = self.cooling_steps // (delta // 5)
            cur_step = self.step // num_steps
            temperature = max(self.initial_temperature - (cur_step * 5), self.final_temperature)

        else:
            raise ValueError(f'Cooling rate {self.cooling} not recognized')

        new_temp = torch.nn.Parameter(torch.log(torch.ones(1) * temperature), requires_grad=False)
        self.model.logit_scale = new_temp
        self.step += 1
        # self.model.logit_scale.to()

    def forward(self, batch):
        image_features = self.encode_image(batch['image'])
        text_features = self.encode_text(batch['text'])
        return image_features, text_features

    def on_train_epoch_start(self):
        if self.lora:
            # manual train mode is needed in order to work with CLIP-LoRA layers
            self.train()
        # self.learnable_parameters()

    def configure_optimizers(self):
        # params = get_lora_parameters(self)
        params = filter(lambda p: p.requires_grad, self.parameters())
        return Adam(params, lr=self.lr)

    def validation_step(self, batch, batch_idx):
        image_features = self.model.encode_image(batch['images'])
        text_features = self.model.encode_text(batch['captions'])

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # adapters
        if hasattr(self, 'vision_adapter'):
            image_features = self.vision_adapter(image_features)

        if hasattr(self, 'text_adapter'):
            text_features = self.text_adapter(text_features)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale.to(image_features.device) * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        ce = torch.nn.CrossEntropyLoss()
        ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=logits_per_image.device)
        self.log('val_loss', (ce(logits_per_image, ground_truth) + ce(logits_per_text, ground_truth)) / 2, sync_dist=True)

        # similarity
        positive_mean = torch.diagonal(logits_per_image).mean()
        off_diagonal = logits_per_image * (1 - torch.eye(logits_per_image.shape[0]).to(logits_per_image.device))
        n = logits_per_image.shape[0]
        negative_mean = off_diagonal.sum() / (n ** 2 - n)
        self.log('mean_positive_similarity', positive_mean, sync_dist=True)
        self.log('mean_negative_similarity', negative_mean, sync_dist=True)

        #retrieval
        targets = torch.eye(logits_per_image.shape[0]).to(logits_per_image.device)
        indexes = torch.arange(targets.shape[0])
        indexes = indexes.repeat(targets.shape[0], 1).T

        targets = torch.eye(logits_per_image.shape[0]).to(logits_per_image.device)
        indexes = torch.arange(targets.shape[0])
        indexes = indexes.repeat(targets.shape[0], 1).T

        for k in [1, 5, 10]:
            rk = RetrievalRecall(top_k=k)
            self.log(f'i2t r@{k}', rk(logits_per_image, targets, indexes), sync_dist=True)
            self.log(f't2i r@{k}', rk(logits_per_image.T, targets, indexes), sync_dist=True)

    def training_step(self, batch, batch_idx):
        if self.cooling is not None:
            self.update_temperature()

        image_features = self.encode_image(batch['images'])
        text_features = self.encode_text(batch['captions'])

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # adapters
        if hasattr(self, 'vision_adapter'):
            image_features = self.vision_adapter(image_features)

        if hasattr(self, 'text_adapter'):
            text_features = self.text_adapter(text_features)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale.to(image_features.device) * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=logits_per_image.device)
        ce = torch.nn.CrossEntropyLoss()
        loss = (ce(logits_per_image, ground_truth) + ce(logits_per_text, ground_truth)) / 2

        optim = self.optimizers()
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

        self.log('train_loss', loss, sync_dist=True)
        self.log('temperature', logit_scale, sync_dist=True)

    def learnable_parameters(self):
        learnable = 0
        total = 0
        for param in self.model.parameters():
            total += param.numel()
            if param.requires_grad:
                learnable += param.numel()

        print(f'total params: {total / 1e6:.2f}M,  learnable params: {learnable / 1e6:.2f}M')
        return total, learnable
