
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import numpy as np

from src.data_preprocessing import create_and_spilt_dataset

class CLIPFineTuner:
    def __init__(self, model_name, device, captions_path, images_folder):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        _, _, self.train_loader, self.val_loader = create_and_spilt_dataset(self.processor.image_processor, self.processor.tokenizer, captions_path, images_folder)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=5e-6)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training", unit="batch"):
            images = batch["image"].to(self.device)
            captions = batch["caption"].to(self.device)
            attention_mask = np.squeeze(batch["attention_mask"]).to(self.device)

            outputs = self.model(pixel_values=images, input_ids=captions, attention_mask=attention_mask)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            labels = torch.arange(images.size(0), device=self.device)

            loss = (self.criterion(logits_per_image, labels) + self.criterion(logits_per_text, labels)) / 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", unit="batch"):
                images = batch["image"].to(self.device)
                captions = np.squeeze(batch["caption"]).to(self.device)
                attention_mask = np.squeeze(batch["attention_mask"]).to(self.device)

                outputs = self.model(pixel_values=images, input_ids=captions, attention_mask=attention_mask)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text

                labels = torch.arange(images.size(0), device=self.device)

                loss = (self.criterion(logits_per_image, labels) + self.criterion(logits_per_text, labels)) / 2

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
