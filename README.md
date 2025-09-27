
# Video Retrieval using Fine-Tuned Persian CLIP

This project is a deep learning model for retrieving videos based on a Persian text query. It uses a fine-tuned CLIP model to find the most relevant videos from a collection. The project includes a web API and a user interface for interacting with the model.

## Project Structure

```
/Project/
├── .gitignore
├── README.md
├── api/
│   ├── app.py
│   └── uploaded_videos/
├── app_ui.py
├── data/
│   ├── captions_new_30k_full.txt
│   └── captions_new.txt
├── model/
│   └── 30k_e7/
│       ├── fine_tuned_clip_model_30k_e7/
│       └── fine_tuned_clip_processor_30k_e7/
├── notebooks/
│   └── CLIP_Persian_FineTune.ipynb
└── src/
    ├── data_preprocessing.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```

## Model

The model is a fine-tuned version of OpenAI's CLIP model. It was fine-tuned on the Flicker30k dataset with Persian captions. The fine-tuning process was done in Google Colab, and the resulting model is saved in the `model/` directory.

## Dataset

The project uses the Flicker30k dataset. The original English captions were translated to Persian for the purpose of this project.

## API and UI

The project uses FastAPI for the backend API and Streamlit for the user interface.

### Running the API

To run the API, navigate to the `api/` directory and run:

```bash
uvicorn app:app --reload
```

### Running the UI

To run the Streamlit UI, run:

```bash
streamlit run app_ui.py
```

## How to Use

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the API and UI:**

    Follow the instructions above to start the API and UI.

3.  **Use the application:**

    Open the Streamlit UI in your browser. You can upload videos and then use a Persian text query to retrieve the most relevant videos.
