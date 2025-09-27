
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import CLIPTokenizer, CLIPProcessor, CLIPImageProcessor
from PIL import Image
from tqdm import tqdm

class ImageData:
    def __init__(self, image_name):
        self.image_name = image_name
        self.captions = []
        self.translated_captions = []

    def add_caption(self, caption):
        self.captions.append(caption)
        self.captions = list(np.squeeze(self.captions))

    def add_translated_caption(self, translated_caption):
        self.translated_captions.append(translated_caption)
        self.translated_captions = list(np.squeeze(self.translated_captions))


class ImageCaption:
    def __init__(self, image_name,caption):
        self.image_name = image_name
        self.caption = caption

class ImageCaptionDataset(Dataset):
    def __init__(self, captions_file, image_dir, image_proccessor, text_tokenizer, image_size=224):
        self.data = []
        self.data_en = []
        self.data_fa = []
        self.image_dir = image_dir
        self.image_proccessor=image_proccessor
        self.text_tokenizer=text_tokenizer

        self._load_data(captions_file)
        self.data = self.data_fa

    def _load_data(self, captions_file):
        with open(captions_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        image_captions = {}

        for i,line in tqdm(enumerate(lines), desc="Processing captions", unit="line"):
            image_name, caption = line.strip().split(",", 1)
            caption = caption.strip(" .\")

            if i%10 > 4:
              image_caption = ImageCaption(image_name, caption)
              self.data_fa.append(image_caption)
            else:
              image_caption = ImageCaption(image_name, caption)
              self.data_en.append(image_caption)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = f"{self.image_dir}/{item.image_name}"
        image = Image.open(image_path).convert("RGB")
        image = self.image_proccessor(images=image, return_tensors="pt")["pixel_values"].squeeze()

        tokenized_caption = self.text_tokenizer(
            item.caption,
            padding="max_length",  # Pad to the maximum length
            truncation=True,
            max_length=64,       # Set a fixed maximum length
            return_tensors="pt"
        )

        return {
            "image_path": image_path,
            "image": image,
            "caption": tokenized_caption["input_ids"],
            "attention_mask": tokenized_caption["attention_mask"],
            "caption_raw": item.caption
        }

def create_and_spilt_dataset(image_processor, text_tokenizer, captions_path, images_folder):
    dataset = ImageCaptionDataset(captions_path, images_folder, image_processor, text_tokenizer, 224)

    batch_size=16
    test_split=0.06

    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dataset, test_dataset, train_loader, test_loader