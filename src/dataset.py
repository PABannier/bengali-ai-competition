import numpy as np
import pandas as pd

import albumentations

import torch

from PIL import Image
import joblib

class BengaliAIDataset(Dataset):
    def __init__(self, folds, img_height, img_width, training=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self.img_height, self.img_width = img_height, img_width

        df = pd.read_csv("../input/train_folds.csv")
        df = df[["image_id", "grapheme_root", "vowel_diacritic", "consonant_diacritic", "kfold"]]

        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values

        
        if training is True:
            self.aug = albumentations.Compose([
                albumentations.Resize(self.img_height, self.img_width, always_apply=True),
                albumentations.Blur(p=0.9),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.1,
                                                rotate_limit=5,
                                                p=0.9),
                albumentations.Normalize(mean, std, always_apply=True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(self.img_height, self.img_width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True)
            ])
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = joblib.load(f"../input/image_pickles/{self.image_ids[idx]}.pkl")
        image = image.reshape(self.img_height, self.img_width).astype(float)
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(np.array(image))["image"]
        image = np.tranpose(image, (2, 0, 1)).astype(np.float32)

        if self.training is True:
            return {
                'image': torch.tensor(image, dtype=torch.float),
                'grapheme_root': torch.tensor(self.grapheme_root, dtype=torch.long),
                'vowel_diacritic': torch.tensor(self.vowel_diacritic, dtype=torch.long),
                'consonant_diacritic':  torch.tensor(self.consonant_diacritic, dtype=torch.long)
            }
        else:
            return {
                'image': torch.tensor(image, dtype=torch.float)
            }









