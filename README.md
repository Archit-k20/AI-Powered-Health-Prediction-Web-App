````markdown
# AI Doctor Assistant — Agentic Health Prediction System

A production-ready, full-stack AI application for smart, interactive preliminary health analysis. Users can enter their symptoms and upload lab reports for instant ML-based disease predictions. The system features an Agentic Triage workflow, a local RAG knowledge base, and SHAP explainability — all fully containerized using Docker.

---

## 🚀 Key Features

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
```text
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
````

-----

## 🖼️ Screenshots

## **Landing Page**

\<img width="3167" height="1466" alt="image" src="https://github.com/user-attachments/assets/a2e681e2-41de-4e33-a0b2-a76f7e45bd05" /\>

## **AI Analysis Tool**

\<img width="3170" height="1729" alt="image" src="https://github.com/user-attachments/assets/7d963fd9-8b1e-4acd-bf6b-e244903d404e" /\>

## **Rest of the Webpage**

\<img width="3166" height="1730" alt="image" src="https://github.com/user-attachments/assets/028bd6a4-928b-4fd8-9597-b9c8fd725372" /\>

## **Results Example**

\<img width="3168" height="1727" alt="image" src="https://github.com/user-attachments/assets/b22b3dd9-0bc1-42bc-a41b-e33ded41dc4d" /\>
\<img width="3169" height="1193" alt="image" src="https://github.com/user-attachments/assets/a10a8c88-8d8d-4332-be37-fd203fd8c0b6" /\>
\<img width="3164" height="1412" alt="image" src="https://github.com/user-attachments/assets/d3be615c-76ac-4838-af99-fb6e34210ee9" /\>

-----

## 🔧 Setup & Usage

### Method 1: Docker (Recommended)

The easiest way to run the application is via Docker Compose, which handles all system dependencies (like Tesseract OCR and spaCy models) automatically.

1.  **Clone the repo:**

    ```bash
    git clone [https://github.com/Lamstersickness/ai-doctor-assistant.git](https://github.com/Lamstersickness/ai-doctor-assistant.git)
    cd ai-doctor-assistant
    ```

2.  **Set up Environment Variables:**
    Create a `.env` file inside the `backend/` folder and add your Gemini API key:

    ```text
    GEMINI_API_KEY=your_actual_api_key_here
    ```

3.  **Build and Run the Containers:**

    ```bash
    docker-compose up --build
    ```

4.  Open your browser and navigate to `http://localhost`.

-----

### Method 2: Local Virtual Environment

If you prefer to run the application without Docker:

1.  **Clone the repo and navigate to it.**
2.  **Set up the `.env` file** in the `backend/` folder as shown above.
3.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
4.  **Install Dependencies & Language Models:**
    ```bash
    cd backend
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
5.  **Start the FastAPI Backend:**
    ```bash
    uvicorn app:app --reload
    ```
6.  **Open the Frontend:**
    Directly open `frontend/index.html` in your browser.

-----

## 📝 Notebooks

  - `01_load_data.ipynb` - Load, clean, and explore raw data
  - `02_preprocess_data.ipynb` - Feature engineering and dataset construction
  - `03_train_models.ipynb` - Build and evaluate ML disease prediction models
  - `04_model_explainability.ipynb` - SHAP and explainability analysis

-----

## 🤖 Tech Stack

  - **Frontend:** HTML, TailwindCSS, JavaScript, Nginx
  - **Backend:** Python, FastAPI, Docker, Docker Compose
  - **AI/LLM Integration:** Gemini 2.0 Flash API (google-genai)
  - **RAG & Vector Search:** LangChain, FAISS, Hugging Face (`all-MiniLM-L6-v2`)
  - **Machine Learning:** scikit-learn (Random Forest), pandas
  - **Explainability:** SHAP
  - **OCR/NLP:** Pytesseract, spaCy

-----

## 📄 License

[MIT](https://www.google.com/search?q=LICENSE)

```
```
