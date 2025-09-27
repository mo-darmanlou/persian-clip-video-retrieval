
import torch
from src.model import CLIPFineTuner

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "openai/clip-vit-base-patch32"
    captions_path = "data/captions_new_30k.txt"
    images_folder = "/home/mohammad/Documents/uni/deeplearning/Project/test/general"

    fine_tuner = CLIPFineTuner(model_name, device, captions_path, images_folder)

    epochs = 7
    fine_tuner.train(epochs)

    # Save the fine-tuned model
    fine_tuner.model.save_pretrained("model/30k_e7/fine_tuned_clip_model_30k_e7")
    fine_tuner.processor.save_pretrained("model/30k_e7/fine_tuned_clip_processor_30k_e7")

if __name__ == "__main__":
    main()
