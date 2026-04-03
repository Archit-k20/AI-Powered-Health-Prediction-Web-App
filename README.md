# AI Doctor Assistant — Smart Health Prediction Web App



A full-stack AI-powered web application for smart, interactive preliminary health analysis. Users can enter their symptoms, personal info, and upload lab reports for instant ML-based disease predictions — all with a modern frontend and a FastAPI backend.



---



## 🚀 Features



- **🤖 Agentic Triage Workflow**
  - Uses Gemini 2.0 Flash to dynamically generate relevant follow-up questions based on initial symptoms.
  - Synthesizes user answers with ML predictions to provide highly personalized health assessments.
  
- **🧠 Retrieval-Augmented Generation (RAG)**
  - Integrated local FAISS vector database powered by Hugging Face embeddings (`all-MiniLM-L6-v2`).
  - Grounds ML disease predictions in verified medical literature to provide actionable precautions.

- **📄 LLM-Powered Lab Report Analyzer**
  - Extracts structured medical data from PDFs and scanned images using the Gemini Vision API.
  - Features **Graceful Degradation**: Automatically falls back to a robust local Regex extraction pipeline if API rate limits are reached.
  
- **📊 Explainable Machine Learning (XAI)**
  - End-to-end ML pipeline (Random Forest Classifier) trained on real-world health data (~85% accuracy).
  - Integrates SHAP (SHapley Additive exPlanations) to transparently show "Why this prediction was made."

- **🐳 Production-Ready Architecture**
  - Fully containerized microservices using Docker and Nginx.
  - Modern, responsive frontend built with TailwindCSS and Vanilla JS.

---


## 📦 Folder Structure

```

backend/
├── models/             # Trained ML model and feature metadata
├── faiss_index/        # Local Vector Database for RAG
├── app.py              # FastAPI backend server
├── vector_store.py     # LangChain/FAISS indexing logic
├── requirements.txt
├── .env                # API Keys (Not committed)
data/                   # Raw, processed, and lookup CSVs
frontend/               # HTML, JS, CSS for Nginx serving
notebooks/              # Jupyter ML pipeline and SHAP analysis
Dockerfile.backend      # Python/FastAPI environment setup
Dockerfile.frontend     # Nginx web server setup
docker-compose.yml      # Multi-container orchestration



```

---



## 🖼️ Screenshots



<!-- Paste images here or drag-and-drop after first pushing to GitHub -->

## **Landing Page**



<img width="3167" height="1466" alt="image" src="https://github.com/user-attachments/assets/a2e681e2-41de-4e33-a0b2-a76f7e45bd05" />



## **AI Analysis Tool**



<img width="3170" height="1729" alt="image" src="https://github.com/user-attachments/assets/7d963fd9-8b1e-4acd-bf6b-e244903d404e" />



## **Rest of the Webpage**



<img width="3166" height="1730" alt="image" src="https://github.com/user-attachments/assets/028bd6a4-928b-4fd8-9597-b9c8fd725372" />





## **Results Example**



<img width="3168" height="1727" alt="image" src="https://github.com/user-attachments/assets/b22b3dd9-0bc1-42bc-a41b-e33ded41dc4d" />

<img width="3169" height="1193" alt="image" src="https://github.com/user-attachments/assets/a10a8c88-8d8d-4332-be37-fd203fd8c0b6" />

<img width="3164" height="1412" alt="image" src="https://github.com/user-attachments/assets/d3be615c-76ac-4838-af99-fb6e34210ee9" />







---



## 🔧 Setup & Usage



Method 1: Docker (Recommended)
The easiest way to run the application is via Docker Compose, which handles all system dependencies (like Tesseract OCR and spaCy models) automatically.

Clone the repo:

```Bash
git clone [https://github.com/Lamstersickness/ai-doctor-assistant.git](https://github.com/Lamstersickness/ai-doctor-assistant.git)
cd ai-doctor-assistant
Set up Environment Variables:
Create a .env file inside the backend/ folder and add your Gemini API key:
```

Plaintext
GEMINI_API_KEY=your_actual_api_key_here
Build and Run the Containers:

```Bash
docker-compose up --build
Open your browser and navigate to http://localhost.
```
Method 2: Local Virtual Environment
If you prefer to run the application without Docker:

Clone the repo and navigate to it.

Set up the .env file in the backend/ folder as shown above.

Create and activate a virtual environment:

```Bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
Install Dependencies & Language Models:
```


```Bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Start the FastAPI Backend:
```


```Bash
uvicorn app:app --reload
Open the Frontend:
Directly open frontend/index.html in your browser.
```


## 📝 Notebooks



- `01_load_data.ipynb` - Load, clean, and explore raw data

- `02_preprocess_data.ipynb` - Feature engineering and dataset construction

- `03_train_models.ipynb` - Build and evaluate ML disease prediction models

- `04_model_explainability.ipynb` - SHAP and explainability analysis



---



## 🤖 Tech Stack



- **Frontend:** HTML, TailwindCSS, JavaScript, Nginx

- **Backend:** Python, FastAPI, Docker, Docker Compose

- **AI/LLM Integration:** Gemini 2.0 Flash API (google-genai)

- **RAG & Vector Search:** LangChain, FAISS, Hugging Face (all-MiniLM-L6-v2)

- **ML/Explainability:** Random Forest, SHAP, Pandas

- **OCR/NLP:** Pytesseract, spaCy, fuzzywuzzy



---



## 📄 License



[MIT](LICENSE)



---
