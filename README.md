🔍 VerifyAI

AI-Powered Document Verification & Tampering Detection (Fraud Detection)

VerifyAI is an AI-powered document screening prototype designed to identify potential document manipulation using OCR, computer vision, image analysis, and machine learning.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/PyTorch-ML-EE4C2C?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Status-Prototype-orange" alt="Prototype"/>
</p>---

🎯 Problem Statement

Document fraud and digital image manipulation can make it difficult to assess the reliability of documents through visual inspection alone. Manual screening is time-consuming and may overlook suspicious inconsistencies.

💡 Our Solution

VerifyAI combines multiple screening techniques to flag suspicious patterns and generate an interpretable risk assessment for further review.

✨ Key Features

Feature| Description
📄 OCR| Extracts text using PaddleOCR
🖼️ Image Analysis| Processes images using OpenCV and Pillow
🔬 Error Level Analysis| Highlights potential compression inconsistencies
🤖 ML Detection| Uses an EfficientNet-B0-based classifier
✅ Rule-Based Checks| Flags configured text and format inconsistencies
🔁 Duplicate Detection| Identifies potential duplicate documents
📊 Risk Assessment| Combines available indicators into a screening score
📝 Reports & History| Supports reporting and local history where implemented

🗂️ Supported Documents

Document Type| Screening Purpose
Aadhaar Card| Visual and content analysis
PAN Card| Visual and content analysis
Railway Ticket| Text and layout checks
Marksheet| Content and layout checks

«⚠️ VerifyAI is a screening tool, not an official government or institutional verification system.»

⚙️ System Architecture

flowchart TD
    A["📄 Upload Document"] --> B["🖼️ Preprocessing"]
    B --> C["🔎 OCR Text Extraction"]
    B --> D["🔬 Image Analysis / ELA"]
    C --> E["✅ Rule-Based Checks"]
    D --> F["🤖 ML Classification"]
    E --> G["📊 Risk Assessment"]
    F --> G
    G --> H["📝 Screening Report"]
    H --> I["👤 Human Review"]

🛠️ Tech Stack

Technology| Purpose
Python| Core development
Streamlit| Web interface
PaddleOCR| Text extraction
OpenCV, Pillow, NumPy| Image processing
PyTorch, EfficientNet-B0| Machine learning
SQLite| Local history storage
Hashing & pHash| Duplicate detection
ReportLab| PDF reporting, where implemented

🧠 Machine Learning Workflow

flowchart LR
    A["📦 Dataset"] --> B["🏷️ Labels & Preprocessing"]
    B --> C["✂️ Train / Test Split"]
    C --> D["🧠 Model Training"]
    D --> E["📈 Evaluation"]
    E --> F["🚀 Application Integration"]

Evaluation metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

Actual model performance should be reported only after training and independent testing are complete.

🚀 Getting Started

1. Clone the repository

git clone https://github.com/deveshrajput922/VerifyAI.git
cd VerifyAI

2. Create a virtual environment

python -m venv .venv

Activate it:

- Windows: ".venv\Scripts\activate"
- Linux/macOS: "source .venv/bin/activate"

3. Install dependencies

pip install -r requirements.txt

4. Run the application

streamlit run app.py

«📌 Ensure dependency filenames and the application entry point match the files in the repository.»

📊 Evaluation & Reliability

Consideration| Details
Dataset| Kriyam TamperFlow, if used for training
Data Splitting| Keep variants of the same source document in one split
ELA| Heuristic indicator, not proof of forgery
Risk Score| Screening indicator, not a fraud probability
QR Decoding| Does not independently prove authenticity
Final Decision| Requires human or authorized verification

🔮 Future Enhancements

- 🎯 Localize suspicious image regions
- 📈 Improve evaluation on unseen documents
- 🧾 Enhance evidence-based reports
- 🔐 Explore authorized verification integrations
- ⚡ Improve inference speed and usability

👥 Team Code Catalyst

#| Team Member
1| Devesh Chauhan
2| Ashish Jha
3| Sarika
4| Nidhi Kumari

---

<p align="center">
  <b>Built with ❤️ by Team Code Catalyst</b><br/>
  <sub>AI • Computer Vision • Document Screening</sub>
</p>
