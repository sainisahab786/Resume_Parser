import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import PyPDF2

data = pd.read_csv('C:/Users/vedansh/Downloads/Resume/Resume.csv')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

data['cleaned_resume'] = data['Resume_str'].apply(clean_text)

X = data['cleaned_resume']
y = data['Category']

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(X).toarray()

joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'resume_model.pkl')

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

app = FastAPI()

model = joblib.load('resume_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def extract_text_from_pdf(file: UploadFile):
    try:
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading PDF file.")

@app.post("/predict-pdf")
async def predict_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
    resume_text = extract_text_from_pdf(file)
    cleaned_text = clean_text(resume_text)
    features = tfidf.transform([cleaned_text])
    prediction = model.predict(features)[0]
    return {"category": prediction}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resume Classification</title>
        <style>
            body {
                font-family: 'Poppins', Arial, sans-serif;
                text-align: center;
                background: linear-gradient(to bottom right, #4A90E2, #D67EEA);
                color: #333;
                margin: 0;
                padding: 0;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .container {
                background: #fff;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                border-radius: 12px;
                padding: 30px 20px;
                max-width: 400px;
                width: 100%;
                animation: fadeIn 1s ease-in-out;
            }
            h1 {
                font-size: 24px;
                margin-bottom: 20px;
                color: #333;
            }
            input[type="file"] {
                display: block;
                width: 100%;
                margin: 15px 0;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 8px;
                background-color: #f9f9f9;
                font-size: 14px;
            }
            button {
                background: linear-gradient(to right, #6A11CB, #2575FC);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            button:hover {
                transform: translateY(-3px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            #result {
                margin-top: 20px;
                font-size: 16px;
                color: #444;
            }
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Resume for Classification</h1>
            <form id="uploadForm">
                <input type="file" id="file" accept="application/pdf" required>
                <button type="submit">Classify Resume</button>
            </form>
            <div id="result"></div>
        </div>

        <script>
            const form = document.getElementById("uploadForm");
            const resultDiv = document.getElementById("result");

            form.addEventListener("submit", async (e) => {
                e.preventDefault();
                resultDiv.innerHTML = "Classifying... Please wait.";

                const fileInput = document.getElementById("file");
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                try {
                    const response = await fetch("/predict-pdf", {
                        method: "POST",
                        body: formData
                    });

                    if (response.ok) {
                        const data = await response.json();
                        resultDiv.innerHTML = `<p>Predicted Category: <strong>${data.category}</strong></p>`;
                    } else {
                        resultDiv.innerHTML = "<p style='color: red;'>Error: Failed to classify resume.</p>";
                    }
                } catch (error) {
                    console.error(error);
                    resultDiv.innerHTML = "<p style='color: red;'>Error: Could not connect to server.</p>";
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

