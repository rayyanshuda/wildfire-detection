"""Dataset and transforms. One canonical definition shared by every model notebook."""
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IN   = '/kaggle/input/notebooks/rayyanshuda/02-split-and-preprocess'
CSV  = f'{IN}/split_v1.csv'
IMGS = f'{IN}/resized'

MEAN, STD, SIZE = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 160

# Augmentation on train only. No colour jitter here on purpose — it is the
# intervention in the shortcut experiment and must not contaminate the baseline.
train_tf = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

eval_tf = T.Compose([
    T.Resize(SIZE),
    T.CenterCrop(SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

# The shortcut intervention: attacks stock-photo *style* (grading, saturation,
# exposure). Hue held at 0.05 because fire is orange, so rotating hue would destroy
# the signal rather than de-bias the model.
aug_tf = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.05),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class WildfireDataset(Dataset):
    """Reads split_v1.csv; labels fire=1.0, nofire=0.0."""

    def __init__(self, csv_path, img_dir, split, transform):
        df = pd.read_csv(csv_path)
        df = df[df.split == split].reset_index(drop=True)
        self.files      = df.out_name.tolist()
        self.labels     = (df.cls == 'fire').astype('float32').tolist()
        self.img_dir    = img_dir
        self.transform  = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = os.path.join(self.img_dir, self.files[i])
        img  = Image.open(path).convert('RGB')      # 12 images are RGBA
        img  = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.float32)
