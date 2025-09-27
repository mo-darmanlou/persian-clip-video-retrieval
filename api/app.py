import torch
import os
import cv2
import shutil
from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import List
import base64
from io import BytesIO
from pydantic import BaseModel

class QueryRequest(BaseModel):
    text: str

app = FastAPI()

# Load fine-tuned CLIP model
model_name = "/home/mohammad/Documents/uni/deep learning/Project/model/30k_e7/fine_tuned_clip_model_30k_e7"
processor_name = "/home/mohammad/Documents/uni/deep learning/Project/model/30k_e7/fine_tuned_clip_processor_30k_e7"

model = CLIPModel.from_pretrained(model_name).eval()
processor = CLIPProcessor.from_pretrained(processor_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
model.to(device)

# Store processed video frames (frame embeddings + metadata)
video_frames_db = []

# Directory to store uploaded videos
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def encode_frame_base():
    video_frames_db.clear()
    folder_path=UPLOAD_DIR
    videos = os.listdir(folder_path)
    videos.sort()
    videoBase=[]
    for video in videos:
        videoBase.append(folder_path+"/"+video)

    for video_path in tqdm(videoBase, desc="Processing videos"):
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        # print(f"Frame rate: {frame_rate}")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_rate == 0:
                # Convert frame to PIL image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb)

                # Preprocess and encode the frame
                # frame_tensor = image_transform(frame_image).unsqueeze(0).to(device)
                frame_tensor = processor.image_processor(images=frame_image, return_tensors="pt")["pixel_values"].to(device)

                with torch.no_grad():
                    frame_embedding = model.get_image_features(pixel_values=frame_tensor)
                    frame_embedding = frame_embedding / frame_embedding.norm(dim=-1, keepdim=True)  # Normalize

                # Store the encoded frame, frame number, and video path
                video_frames_db.append((frame_embedding.cpu(), frame_count, video_path, frame_image))

            frame_count += 1
        cap.release()
    
    


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video and saves it to disk"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Video uploaded successfully", "video_path": file_path}

@app.post("/process/")
def process_videos():
    """Extracts frames, encodes them, and stores embeddings"""
    encode_frame_base()
    return {"message": "Frames processed and stored"}

@app.post("/query/")
def query_text(request: QueryRequest):
    """Encodes input text and finds the most similar frames"""
    # text_inputs = processor.text([text], return_tensors="pt", padding=True, truncation=True).to(device)

    # with torch.no_grad():
    #     text_embedding = model.get_text_features(**text_inputs)
    #     text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    # similarities = []
    # for frame in video_frames_db:
    #     similarity = torch.cosine_similarity(text_embedding, frame["embedding"], dim=-1).item()
    #     similarities.append((frame["video_path"], frame["frame_number"], similarity))

    # top_matches = sorted(similarities, key=lambda x: x[2], reverse=True)[:5]

    # return {"top_matches": top_matches}

    text_inputs = processor(text=[request.text], images=None, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # Normalize embedding


    # Compute similarities
    similarities = []
    for frame_embedding, frame_number, video_path, frame in video_frames_db:
        frame_embedding = frame_embedding.to(device)
        similarity = torch.cosine_similarity(text_embedding, frame_embedding, dim=-1).item()
        similarities.append((similarity, frame_number, video_path, frame))


    sorted_sims = sorted(similarities, key=lambda x: x[0], reverse=True)
    # print("sorted_sims",sorted_sims)

    sorted_video_paths = {}
    for similarity, frame_number, video_path, frame in sorted_sims:
        if video_path not in sorted_video_paths.keys():
            sorted_video_paths[video_path] = similarity ** (1/4)

    j = {"video_similarities": sorted_video_paths}
    print("json: ",j)
    return j


    # # Sort by similarity and get top 5 matches
    # top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:5]
    

    # # Print the results
    # print("Top 5 Matches:")
    # image_results = []
    # for similarity, frame_number, video_path, frame in top_matches:
    #     print(f"Video: {video_path}, Frame: {frame_number}, Similarity: {similarity:.4f}")
    #     # cap = cv2.VideoCapture(video_path)
    #     # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    #     # ret, frame = cap.read()
    #     # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #     buffered = BytesIO()
    #     frame.save(buffered, format="JPEG")
    #     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    #     image_results.append({
    #         "video_path": video_path,
    #         "frame_number": frame_number,
    #         "similarity": similarity,
    #         "image_base64": img_str
    #     })

    # j = {"top_matches": image_results}
    # # print("json: ",j)
    # return j


# @app.post("/query/")
# def query_text(text: str):
#     """Encodes input text and finds the most similar frames"""
#     text_inputs = processor.text([text], return_tensors="pt", padding=True, truncation=True).to(device)

#     with torch.no_grad():
#         text_embedding = model.get_text_features(**text_inputs)
#         text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

#     similarities = []
#     for frame in video_frames_db:
#         similarity = torch.cosine_similarity(text_embedding, frame["embedding"], dim=-1).item()
#         similarities.append((frame["video_path"], frame["frame_number"], similarity, frame["image"]))

#     top_matches = sorted(similarities, key=lambda x: x[2], reverse=True)[:5]
    
#     # Convert top 5 frames to Base64 for easier UI display
#     from io import BytesIO
#     import base64
    
#     image_results = []
#     for video_path, frame_number, similarity, image in top_matches:
#         buffered = BytesIO()
#         image.save(buffered, format="JPEG")
#         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         image_results.append({
#             "video_path": video_path,
#             "frame_number": frame_number,
#             "similarity": similarity,
#             "image_base64": img_str
#         })

#     return {"top_matches": image_results}

@app.get("/")
def root():
    return {"message": "CLIP Video Retrieval API Running"}


