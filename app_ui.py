import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Video Retrieval System")

# Upload Video
st.header("Upload a Video")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    files = {"file": open("temp_video.mp4", "rb")}
    response = requests.post(f"{API_URL}/upload/", files=files)
    st.write(response.json())

# Process Video
st.header("Process Video Frames")
video_path = st.text_input("Enter video path to process", "")
if st.button("Process Frames"):
    response = requests.post(f"{API_URL}/process/", json={"video_path": video_path})
    st.write(response.json())

# Query Text
st.header("Search for Relevant Frames")
query = st.text_input("Enter search query")
if st.button("Retrieve Frames"):
    response = requests.post(f"{API_URL}/query/", json={"text": query})
    results = response.json()
    st.write("Top Retrieved Frames:")
    for match in results["top_matches"]:
        st.write(f"Video: {match[0]}, Frame: {match[1]}, Similarity: {match[2]:.4f}")

