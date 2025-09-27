
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

def encode_frame_base(model, test_loader, device):
    model.eval()
    frameBaseEncoded = []
    paths = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="encoding frameBase", unit="batch"):
            image_paths = batch["image_path"]
            images = batch["image"].to(device)

            image_embeddings = model.get_image_features(pixel_values=images)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

            for path,embedding in zip(image_paths, image_embeddings.cpu()):
                if path not in paths:
                  paths.append(path)
                  frameBaseEncoded.append((path, embedding))
    print("frameBaseEncoded",len(frameBaseEncoded))
    return frameBaseEncoded


def inference(frameBaseEncoded, text, model, processor, device):
    text_inputs = processor(text=[text], images=None, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    similarities = []
    for image_path, image_embedding in frameBaseEncoded:
        image_embedding = image_embedding.to(device)
        similarity = torch.cosine_similarity(text_embedding, image_embedding, dim=-1).item()
        similarities.append((image_path, similarity))

    top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    print("Top 5 Matches:")
    for image_path, similarity in top_matches:
        print(f"Image: {image_path}, Similarity: {similarity:.4f}")
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.show()

def AUC_score(model, processor, test_loader, device):
    model.eval()

    subset_dataset = Subset(test_loader.dataset, range(200))
    subset_loader = DataLoader(subset_dataset, batch_size=16, shuffle=False, num_workers=2)

    frameBaseEncoded = []
    paths = []
    frameCaptions = []
    with torch.no_grad():
        for batch in tqdm(subset_loader, desc="encoding frameBase", unit="batch"):
            image_paths = batch["image_path"]
            images = batch["image"].to(device)
            captions = np.squeeze(batch["caption_raw"])

            image_embeddings = model.get_image_features(pixel_values=images)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

            for path,embedding in zip(image_paths, image_embeddings.cpu()):
              if path not in paths:
                paths.append(path)
                frameBaseEncoded.append((path, embedding))

            for caption in captions:
                frameCaptions.append((path, caption))


        print("frameCaptions",len(frameCaptions))
        print("frameBaseEncoded",len(frameBaseEncoded))

        AUC_score = 0

        for path,caption in frameCaptions:
            text_inputs = processor(text=[caption], images=None, return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                text_embedding = model.get_text_features(**text_inputs)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            similarities = []
            for image_path, image_embedding in frameBaseEncoded:
                image_embedding = image_embedding.to(device)
                similarity = torch.cosine_similarity(text_embedding, image_embedding, dim=-1).item()
                similarities.append((image_path, similarity))

            sim_scores = [similarity for path,similarity in similarities]

            ground_truth = np.zeros_like(sim_scores)
            for i, (image_path, _) in enumerate(similarities):
                if image_path == path:
                    ground_truth[i] = 1

            score = roc_auc_score(ground_truth, sim_scores)
            AUC_score += score

        AUC_score /= len(frameCaptions)
        print("AUC score: ",AUC_score)
        return AUC_score
