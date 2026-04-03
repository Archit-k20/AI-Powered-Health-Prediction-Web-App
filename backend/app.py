from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from google import genai
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np
import json
import os
import io
from typing import List
from fastapi import Query
from fuzzywuzzy import process
from fastapi import Body
from fastapi import UploadFile, File
import re
import spacy
from textblob import TextBlob
import pytesseract
from PIL import Image
import pdfplumber
import shap
from vector_store import get_medical_context

ner_nlp = spacy.load("en_core_web_sm")
def humanize_feature_name(feat):
    return feat.replace('_', ' ').replace('-', ' ').capitalize()

app = FastAPI()

# Load environment variables
load_dotenv()

# Configure Gemini API for Lab Analysis using the NEW SDK
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configured successfully for lab analysis")
    except Exception as e:
        print(f"❌ Failed to configure Gemini API: {e}")
else:
    print("⚠️ Warning: GEMINI_API_KEY not found in .env file. Smart lab analysis will be disabled.")

# Tightened CORS | Removed wildcard '*'
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Wildcard allows file:/// protocol to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled Server Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", "disease_prediction_model.pkl")
label_mapping_path = os.path.join(current_dir, "models", "label_mapping.json")
feature_names_path = os.path.join(current_dir, "models", "feature_names.csv")

print(f"\nModel path: {model_path}")
print(f"Label mapping path: {label_mapping_path}")
print(f"Features path: {feature_names_path}")
# Initialize variables safely
model = None
label_mapping = {}
features = []
shap_explainer = None

try:
    print("\nLoading model files...")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        shap_explainer = shap.TreeExplainer(model)
        print("✅ SHAP explainer loaded")
    else:
        print(f"⚠️ Warning: Model file not found at {model_path}")
    
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path) as f:
            label_mapping = json.load(f)
        print(f"✅ Loaded {len(label_mapping)} diseases")
        
    features_path = os.path.join(current_dir, "../data/X_processed.csv")
    if os.path.exists(features_path):
        X = pd.read_csv(features_path)
        features = list(X.columns)
        print(f"✅ Loaded {len(features)} features")
        
except Exception as e:
    print(f"❌ Error loading model files (Server boot continuing): {e}")
try:
    shap_explainer = shap.TreeExplainer(model)
    print("✅ SHAP explainer loaded")
except Exception as e:
    shap_explainer = None
    print("❌ SHAP explainer error: ", e)

synonym_file = os.path.join(current_dir, "../data/symptom_synonyms.json")
with open(synonym_file) as f:
    synonym_dict = json.load(f)
symptom_terms = []
term_to_main = {}
for main, syns in synonym_dict.items():
    symptom_terms.append(main)
    term_to_main[main.lower()] = main
    for syn in syns:
        symptom_terms.append(syn)
        term_to_main[syn.lower()] = main

print(f"\nLoaded {len(term_to_main)} symptom mappings")

frontend_path = os.path.abspath(os.path.join(current_dir, "..", "frontend"))
print(f"\nLooking for frontend at: {frontend_path}")

if os.path.exists(frontend_path):
    print("✅ Found frontend directory")
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"❌ Frontend directory not found at: {frontend_path}")
    print("Current directory structure:")
    print(os.listdir(os.path.dirname(current_dir)))

@app.options("/predict")
async def predict_options():
    return {"status": "ok"}

@app.post("/agent_triage")
async def agent_triage(symptoms: str = Form(...)):
    if gemini_client is None:
        return {"questions": []}
        
    try:
        sym_list = json.loads(symptoms)
        if not sym_list:
            return {"questions": []}
            
        prompt = f"The patient reports these symptoms: {', '.join(sym_list)}. Ask 2 specific follow-up questions to narrow down the diagnosis. Output strict JSON with a single key 'questions' containing an array of 2 strings."
        
        # Using Gemini to generate the follow-up questions
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'system_instruction': 'You are a medical triage AI. Output strict JSON.'
            }
        )
        data = json.loads(response.text.strip())
        return {"questions": data.get("questions", [])[:2]}
    except Exception as e:
        print(f"Agent Triage Error: {e}")
        return {"questions": []}

@app.post("/predict")
async def predict_disease(
    symptoms: str = Form(...),
    age: str = Form(None),
    gender: str = Form(None),
    weight: str = Form(None),
    height: str = Form(None),
    agent_answers: str = Form(None), # <-- Added this parameter
    lab_reports: List[UploadFile] = File(None)
):
    # Safety check added here
    if model is None or not features:
        raise HTTPException(status_code=503, detail="The AI prediction model is currently unavailable on the server.")
    
    print(f"Received request: POST /predict")
    try:
        # Parse the symptoms JSON string
        symptoms_data = json.loads(symptoms)
        print("Received symptoms:", symptoms_data)
        
        if not isinstance(symptoms_data, list):
            raise HTTPException(400, "Invalid input format - 'symptoms' must be a JSON list")
        
        print("Processing symptoms:", symptoms_data)
        
        clean_features = [f.strip() for f in features]
        backend_symptoms = []
        unmatched = []
        
        for symptom in symptoms_data:
            mapped = term_to_main.get(symptom)
            converted = symptom.lower().replace(" ", "_")
            
            if mapped and mapped in clean_features:
                backend_symptoms.append(mapped)
            elif converted in clean_features:
                backend_symptoms.append(converted)
            else:
                unmatched.append(symptom)
        
        print("Mapped symptoms:", backend_symptoms)
        if unmatched:
            print("Unmapped symptoms:", unmatched)

        input_vector = [1 if symptom in backend_symptoms else 0 for symptom in clean_features]
        probabilities = model.predict_proba([input_vector])[0]
        
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > 0.01:
                predictions.append({
                    "disease": label_mapping[str(i)],
                    "probability": round(float(prob) * 100, 2)
                })
        
        predictions.sort(key=lambda x: x["probability"], reverse=True)

        explanation_text = ""
        top_features = []

        if shap_explainer:
            shap_vals = shap_explainer.shap_values(np.array([input_vector]))
            pred_class_idx = None
            if predictions and predictions[0]['disease']:
                for k, v in label_mapping.items():
                    if v == predictions[0]['disease']:
                        pred_class_idx = int(k)
                        break

            print("SHAP debugging info:")
            print("  shap_vals type:", type(shap_vals))
            if isinstance(shap_vals, list):
                print("  shap_vals len:", len(shap_vals))
                for idx, arr in enumerate(shap_vals):
                    print(f"  shap_vals[{idx}] shape:", np.shape(arr))
            else:
                print("  shap_vals shape:", np.shape(shap_vals))
            print("  pred_class_idx:", pred_class_idx)

            vals = None
            # Safe selection for multiclass OR binary SHAP output:
            if isinstance(shap_vals, list) and pred_class_idx is not None and pred_class_idx < len(shap_vals):
                vals = shap_vals[pred_class_idx][0]
            elif isinstance(shap_vals, list) and len(shap_vals) == 1:
                vals = shap_vals[0][0]  # single-class
            elif isinstance(shap_vals, np.ndarray):
                vals = shap_vals[0]
            else:
                explanation_text = "Model explanation is unavailable (internal)"
            
            def to_scalar(x):
                if isinstance(x, np.ndarray):
                    # If array has more than 1 element, just take the first (or fallback 0)
                    if x.size == 1:
                        return x.item()
                    else:
                        return float(x.flat[0])
                if isinstance(x, list):
                    return x[0]
                return x

            if vals is not None:
                # Get top positive (important) features
                top_idx = vals.argsort()[-3:][::-1].tolist()
                if top_idx and isinstance(top_idx[0], list):  # flatten if needed
                    top_idx = [item for sublist in top_idx for item in sublist]
                top_features = [
                    features[int(i)]
                    for i in top_idx
                    if to_scalar(input_vector[int(i)]) == 1 and to_scalar(vals[int(i)]) > 0
                ]
                if not top_features:
                    top_features = [features[int(i)] for i in top_idx]
                if top_features:
                    feat_nice = [humanize_feature_name(f) for f in top_features[:3]]
                    explanation_text = (
                        f"Most important symptoms for this prediction: "
                        + ", ".join(feat_nice)
                        + "."
                    )   
                else:
                    explanation_text = (
                        f"The prediction for <b>{predictions[0]['disease']}</b> was based on your provided symptoms."
                    )
            else:
                explanation_text = "Explanation is not available due to a server limitation."
        # --- RAG INTEGRATION ---
        recommendations = []
        if predictions:
            primary_disease = predictions[0]["disease"]
            # Query FAISS vector store (Local HuggingFace)
            medical_context = get_medical_context(primary_disease)
            predictions[0]["description"] = medical_context["description"]
            recommendations = medical_context["precautions"]

        # --- AGENTIC SYNTHESIS (GEMINI) ---
        if gemini_client and agent_answers:
            try:
                answers_dict = json.loads(agent_answers)
                agent_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in answers_dict.items()])
                
                synthesis_prompt = f"""
                ML Prediction: {primary_disease}
                ML Reasoning: {explanation_text}
                
                Patient answers to follow-ups:
                {agent_context}
                
                Write a comforting 2-sentence explanation combining the ML reasoning and the patient's specific answers.
                """
                
                res = gemini_client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=synthesis_prompt,
                    config={
                        'system_instruction': 'You are an empathetic AI doctor.'
                    }
                )
                explanation_text = res.text.strip()
            except Exception as e:
                print(f"Agent synthesis failed: {e}")

        return {
            "most_likely": predictions[0] if predictions else None,
            "possible": predictions[1:4],
            "matched_symptoms": backend_symptoms,
            "explanation": explanation_text,
            "recommendations": recommendations
        }
        
    except json.JSONDecodeError as e:
        print("JSON decode error:", str(e))
        raise HTTPException(400, detail="Invalid JSON format for 'symptoms'")
    except Exception as e:
        print("Prediction error:", str(e))
        raise HTTPException(400, detail=str(e))

@app.get("/symptom_suggest")
def symptom_suggest(query: str = Query(...)):
    if not query: return {"suggestions": []}
    results = process.extractBests(query, symptom_terms, limit=6, score_cutoff=60)
    mains = []
    for res, score in results:
        canonical = term_to_main.get(res.lower(), res)
        if canonical not in mains:
            mains.append(canonical)
    return {"suggestions": mains[:5]}

@app.post("/extract_entities")
async def extract_entities(payload: dict = Body(...)):
    text = payload.get("text", "")
    doc = ner_nlp(text)
    symptoms, body_parts = [], []
    duration, sentiment = "", ""
    # Basic NER (expand labels for better results)
    for ent in doc.ents:
        if ent.label_ in ["SYMPTOM", "DISEASE", "NORP", "EVENT"]:
            symptoms.append(ent.text)
        if ent.label_ in ["ORG", "GPE", "LOC", "FAC"]:
            body_parts.append(ent.text)
        if ent.label_ == "DATE":
            duration = ent.text
    # Fallback: keyword search
    if not symptoms:
        for tok in doc:
            if any(word in tok.lemma_.lower() for word in ["ache", "pain", "cough", "fever", "tired", "fatigue", "nausea", "dizzy", "headache"]):
                symptoms.append(tok.text)

    # After/inside fallback symptom chunk extraction
    for chunk in doc.noun_chunks:
        text = chunk.text.lower().strip()
        # Fuzzy match against all symptoms
        best, score = process.extractOne(text, list(term_to_main.keys()))
        if score >= 85:   # or use 70 for more forgiving
            symptoms.append(term_to_main[best])

    # TextBlob sentiment
    try:
        sentpol = TextBlob(text).sentiment.polarity
        if sentpol < -0.2: sentiment = "Negative"
        elif sentpol > 0.2: sentiment = "Positive"
        else: sentiment = "Neutral"
    except:
        sentiment = "Neutral"
    return {"symptoms": list(set(symptoms)), "body_parts": list(set(body_parts)), "duration": duration, "sentiment": sentiment}

def ocr_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image)
    return text

def extract_text_pdf(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        texts = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return "\n".join(texts)
def fallback_regex_extraction(text):
    """Fallback method if the LLM API fails due to rate limits."""
    pattern = re.compile(r'([A-Za-z \(\)\-/]+)\s*[:\-]?\s*([\d\.]+)\s*([^\s\d]+)?(?:.*?(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*)?', re.I)
    findings = []
    for match in pattern.findall(text):
        test, value, units, ref_lo, ref_hi = match
        if test and value:
            findings.append({
                "test_name": test.strip(),
                "value": value,
                "units": units.strip() if units else "",
                "reference_range": f"{ref_lo}-{ref_hi}" if ref_lo and ref_hi else ""
            })
            
    summary = "⚙️ Basic System Extraction (API Quota Reached):\n" + "-"*40 + "\n"
    for f in findings:
        summary += f"• {f['test_name']}: {f['value']} {f['units']} (Ref: {f['reference_range']})\n"
        
    return {"findings": findings, "summary": summary, "raw": text[:500]}
@app.post("/analyze_lab_report")
async def analyze_lab_report(file: UploadFile = File(...)):
    if gemini_client is None:
        return {"error": "LLM Analysis is not configured. Please check the server's OpenAI API key."}

    ext = file.filename.lower().split('.')[-1]
    raw_bytes = await file.read()
    
    # Step 1: Extract raw text via OCR or PDF parsing
    try:
        if ext in ["jpg", "jpeg", "png"]:
            text = ocr_image(raw_bytes)
        elif ext == "pdf":
            text = extract_text_pdf(raw_bytes)
        else:
            return {"error": "Unsupported file type. Please upload a PDF or Image."}
            
        if not text.strip():
            return {"error": "Could not extract any readable text from the file."}
            
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

    # Step 2: Use OpenAI to intelligently structure the text
    prompt = f"""
    Analyze the following raw OCR text from a medical lab report. 
    Extract the test results and return a JSON object with a single key named "findings". 
    The value of "findings" must be an array of objects, where each object has these exact keys: "test_name", "value", "units", "reference_range".
    If a specific piece of information is missing for a test, leave the string empty "".
    
    Raw Lab Report Text:
    {text[:5000]}
    """

    try:
        # Generate content using the new Gemini SDK
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        response_text = response.text.strip()
        
        # Parse the JSON string
        parsed_data = json.loads(response_text)
        findings = parsed_data.get("findings", [])
        
        # Format a nice summary for the frontend
        summary = "🩺 Smart AI Lab Extraction (OpenAI):\n" + "-"*35 + "\n"
        for f in findings:
            test = f.get('test_name', 'Unknown Test')
            val = f.get('value', 'N/A')
            unit = f.get('units', '')
            ref = f.get('reference_range', 'No ref')
            summary += f"• {test}: {val} {unit} (Ref: {ref})\n"
            
        return {"findings": findings, "summary": summary, "raw": text[:500]}
        
    except json.JSONDecodeError:
        return {"error": "AI successfully analyzed the document but returned unparseable formatting. Please try again."}
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Quota" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {"error": "⚠️ The AI is currently processing too many requests. Please wait 1 minute and try again."}
        return {"error": f"AI extraction failed: {error_msg}"}
    

if __name__ == "__main__":
    import uvicorn
    print("\nStarting server...")
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="debug"
    )