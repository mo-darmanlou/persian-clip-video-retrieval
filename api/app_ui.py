import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

API_URL = "http://127.0.0.1:8000"

st.title("Video Retrieval System")

# Upload Multiple Videos
st.header("Upload Videos")
uploaded_files = st.file_uploader("Choose video files", type=["mp4", "avi", "mov"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"  # Temporary file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        files = {"file": open(file_path, "rb")}
        response = requests.post(f"{API_URL}/upload/", files=files)
        st.write(response.json())

# Process Video
st.header("Process Video Frames")
if st.button("Process Frames"):
    response = requests.post(f"{API_URL}/process/")
    st.write(response.json())

# Query Text
# st.header("Search for Relevant Frames")
# query = st.text_input("Enter search query")
# if st.button("Retrieve Frames"):
#     response = requests.post(f"{API_URL}/query/", json={"text": query})
#     # response = requests.post(f"{API_URL}/query/", data=query, headers={'Content-Type': 'text/plain'})
#     results = response.json()
#     st.write("Top Retrieved Frames:", results)
    
#     for match in results["top_matches"]:
#         st.write(f"Video: {match['video_path']}, Frame: {match['frame_number']}, Similarity: {match['similarity']:.4f}")
#         img_data = base64.b64decode(match["image_base64"])
#         image = Image.open(BytesIO(img_data))
#         st.image(image, caption=f"Frame {match['frame_number']} from {match['video_path']}", use_column_width=True)


# Query Text
st.header("Search for Relevant Videos")
query = st.text_input("Enter search query")
if st.button("Retrieve Videos"):
    response = requests.post(f"{API_URL}/query/", json={"text": query})
    if response.status_code == 200:
        results = response.json()
        st.write("Top Retrieved Videos:")
        
        print("video_similarities", results["video_similarities"])
        video_sims = results["video_similarities"]
        for video_path in results["video_similarities"].keys():
            st.write(f"Video: {video_path}, Similarity: {video_sims[video_path]:.4f}")
            video_url = f"{video_path}"
            st.video(video_url)

    else:
        st.write("Error:", response.text)