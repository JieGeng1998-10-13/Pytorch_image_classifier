from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import functional as F
import torch

class DataSet_V2:
    def __init__(self, root_dir, batch_size, shuffle, num_workers, istrainning):
        super(DataSet_V2, self).__init__()
        self.istrainning = istrainning
        self.dataset = datasets.ImageFolder(root=root_dir,
                                            transform=self.get_transforms())
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else num_workers * batch_size
        )

    def get_transforms(self):
        if not self.istrainning:
            return transforms.Compose([
                transforms.Lambda(self.convert_img),
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224), antialias=True),
            ])
        else:
            return transforms.Compose([
                transforms.Lambda(self.convert_img),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.Lambda(lambda img: self.add_random_noise(img)),
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224), antialias=True),
            ])

    @staticmethod
    def add_random_noise(img):
        """
        向图像添加随机噪声
        :param img: PIL.Image 输入图像
        :return: 含噪声的 PIL.Image 图像
        """
        img_tensor = F.to_tensor(img)
        noise = torch.randn_like(img_tensor) * 0.1
        noisy_img_tensor = img_tensor + noise
        noisy_img_tensor = torch.clamp(noisy_img_tensor, 0, 1)
        return F.to_pil_image(noisy_img_tensor)

    @staticmethod
    def convert_img(img):
        return img.convert("RGB")

    def __len__(self):
        return len(self.dataset.imgs)

    def __iter__(self):
        for data in self.loader:
            yield data

            
if __name__ == '__main__':
    batch_size = 8
    num_workers = 0
    train_dataset = DataSet_V2('./data/train', batch_size, True, num_workers, istrainning=True)
    test_dataset = DataSet_V2('./data/test', batch_size, False, num_workers, istrainning=False)

    for inputs, labels in train_dataset:
        print(labels[0].item())
