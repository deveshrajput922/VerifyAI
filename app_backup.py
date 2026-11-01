import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
import pypdfium2 as pdfium
import cv2
import numpy as np
import tempfile
import os
import json
import re
import hashlib
import io
import xml.etree.ElementTree as ET
from datetime import datetime

# ---------------------------------------------------------
# OCR + screening helpers
# ---------------------------------------------------------

@st.cache_resource
def load_ocr_model():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )


def check_image_quality(pil_image):
    image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = image.shape
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    issues = []
    if width < 600 or height < 400:
        issues.append("Low image resolution")
    if sharpness < 100:
        issues.append("Image may be blurry")
    return width, height, sharpness, issues


def _decode_qr_with_opencv(image):
    detector = cv2.QRCodeDetector()
    # Try normal, enlarged, grayscale and adaptive-threshold versions.
    candidates = [image]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    candidates.append(cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR))
    candidates.append(cv2.cvtColor(
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 5),
        cv2.COLOR_GRAY2BGR
    ))
    for candidate in candidates:
        for scale in (1, 2, 3):
            work = candidate if scale == 1 else cv2.resize(
                candidate, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )
            try:
                data, _, _ = detector.detectAndDecode(work)
                if data:
                    return data
                ok, decoded, _, _ = detector.detectAndDecodeMulti(work)
                if ok and decoded:
                    values = [x for x in decoded if x]
                    if values:
                        return "\n".join(values)
            except cv2.error:
                continue
    return None


def detect_qr(pil_image):
    image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    data = _decode_qr_with_opencv(image)

    # Optional stronger decoder. Install with: python -m pip install zxing-cpp
    if not data:
        try:
            import zxingcpp
            rgb = np.array(pil_image.convert("RGB"))
            results = zxingcpp.read_barcodes(rgb)
            values = [item.text for item in results if item.text]
            if values:
                data = "\n".join(values)
        except ImportError:
            pass
        except Exception:
            pass

    if data:
        st.success("QR code detected and decoded.")
        st.markdown("**Decoded QR data**")
        # Payload may contain personal information. Keep it collapsed by default.
        with st.expander("View decoded QR payload", expanded=False):
            st.code(data)
    else:
        st.info(
            "No readable QR code detected. The document may not contain a supported "
            "QR code, or the QR may be damaged, cropped, or encoded in an unsupported format."
        )
    st.caption("QR detection and decoding do not establish document authenticity.")
    return data or None


def extract_text(pil_image):
    temp_path = None
    try:
        ocr = load_ocr_model()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name
        pil_image.convert("RGB").save(temp_path, format="PNG")
        results = ocr.predict(temp_path)
        text_lines = []
        for item in results or []:
            payload = item
            if hasattr(payload, "json"):
                payload = payload.json
                if callable(payload):
                    payload = payload()
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                continue
            result_data = payload.get("res", payload)
            if isinstance(result_data, dict):
                recognized = result_data.get("rec_texts", [])
                if isinstance(recognized, list):
                    text_lines.extend(
                        str(line).strip() for line in recognized if str(line).strip()
                    )
        return list(dict.fromkeys(text_lines)), None
    except Exception as exc:
        return [], str(exc)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def show_ocr_results(pil_image):
    st.markdown("#### Text extraction (OCR)")
    with st.spinner("Reading text from the document..."):
        text_lines, error = extract_text(pil_image)
    if error:
        st.error("OCR could not process this document.")
        st.caption(f"Technical detail: {error}")
        return [], error
    if text_lines:
        extracted_text = "\n".join(text_lines)
        st.success(f"Recognized {len(text_lines)} text lines.")
        st.text_area(
            "Extracted text",
            value=extracted_text,
            height=250,
            key=f"ocr_{pil_image.width}_{pil_image.height}_{hash(extracted_text)}"
        )
        st.download_button(
            "Download extracted text",
            data=extracted_text,
            file_name="verifyai_extracted_text.txt",
            mime="text/plain"
        )
    else:
        st.warning("No readable text was extracted. Try a clearer image.")
    st.caption("OCR output may contain mistakes. Review it against the original document.")
    return text_lines, None


def classify_document(text):
    t = text.upper()
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", t) or "INCOME TAX" in t or "PERMANENT ACCOUNT NUMBER" in t:
        return "PAN Card"
    if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", t) or "AADHAAR" in t or "आधार" in t:
        return "Aadhaar Card"
    if "DRIVING LICENCE" in t or "DRIVING LICENSE" in t:
        return "Driving Licence"
    if "PASSPORT" in t or "REPUBLIC OF INDIA" in t:
        return "Passport"
    if "MARKSHEET" in t or "STATEMENT OF MARKS" in t or "BOARD OF" in t:
        return "Marksheet / Certificate"
    return "Unknown / Other"


def extract_fields(text):
    fields = {}
    upper = text.upper()
    pan = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", upper)
    aadhaar = re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", text)
    dob = re.search(
        r"\b(?:DOB|DATE OF BIRTH|D.O.B)\s*[:\-]?\s*([0-3]?\d[/-][01]?\d[/-](?:19|20)?\d{2})",
        text, re.IGNORECASE
    )
    if pan:
        fields["PAN number"] = pan.group(0)
    if aadhaar:
        digits = re.sub(r"\D", "", aadhaar.group(0))
        fields["Aadhaar number"] = f"XXXX XXXX {digits[-4:]}" if len(digits) == 12 else "Pattern found; review manually"
    if dob:
        fields["Date of birth"] = dob.group(1)
    # Basic name extraction from common Aadhaar/PAN labels. OCR may misread labels.
    name_patterns = [
        r"(?:NAME|नाम)\s*[:\-]?\s*([A-Z][A-Z .'-]{2,})",
        r"(?:FATHER'?S NAME|FATHER NAME)\s*[:\-]?\s*([A-Z][A-Z .'-]{2,})"
    ]
    for pattern in name_patterns:
        match = re.search(pattern, upper)
        if match:
            candidate = re.split(
                r"\b(?:GOVERNMENT|GOVT|INDIA|DATE|DOB|FATHER|MALE|FEMALE)\b",
                match.group(1)
            )[0].strip(" .-")
            if candidate:
                fields["Name"] = candidate.title()
                break
    return fields


def validate_fields(text, doc_type):
    upper = text.upper()
    checks = []
    if doc_type == "PAN Card":
        ok = bool(re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", upper))
        checks.append(("PAN format", ok, "PAN-like pattern found" if ok else "PAN pattern not found"))
    elif doc_type == "Aadhaar Card":
        ok = bool(re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text))
        checks.append(("Aadhaar format", ok, "12-digit pattern found; not an official validity check" if ok else "12-digit pattern not found"))
    elif doc_type == "Passport":
        ok = "PASSPORT" in upper
        checks.append(("Passport keyword", ok, "Keyword found" if ok else "Keyword not found"))
    else:
        checks.append(("Document type", False, "Could not confidently identify document type"))
    return checks


def image_manipulation_indicator(pil_image):
    original = pil_image.convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    diff = ImageChops.difference(original, recompressed)
    extrema = diff.getextrema()
    mean_diff = sum((lo + hi) / 2 for lo, hi in extrema) / 3
    return float(mean_diff), ImageEnhance.Contrast(diff).enhance(4)


# ---------------------------------------------------------
# QR payload parsing + OCR/QR field comparison
# ---------------------------------------------------------

def _flatten_payload(obj, output):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                _flatten_payload(value, output)
            else:
                output[str(key).strip().lower()] = str(value).strip()
    elif isinstance(obj, list):
        for item in obj:
            _flatten_payload(item, output)


def parse_qr_payload(payload):
    """Parse common JSON, XML, or key:value QR payloads. Unknown/binary formats stay unsupported."""
    if not payload:
        return {}, "No QR payload available."
    raw = payload.strip()
    parsed = {}

    try:
        obj = json.loads(raw)
        _flatten_payload(obj, parsed)
    except Exception:
        pass

    if not parsed and raw.startswith("<"):
        try:
            root = ET.fromstring(raw)
            for node in root.iter():
                if node.text and node.text.strip():
                    parsed[node.tag.split("}")[-1].strip().lower()] = node.text.strip()
                for key, value in node.attrib.items():
                    parsed[key.strip().lower()] = str(value).strip()
        except Exception:
            pass

    if not parsed:
        for line in re.split(r"[\r\n|;,]+", raw):
            match = re.match(r"\s*([A-Za-z_ ]{2,30})\s*[:=]\s*(.*?)\s*$", line)
            if match and match.group(2):
                parsed[match.group(1).strip().lower()] = match.group(2).strip()

    if not parsed:
        return {}, (
            "QR was decoded, but its payload format could not be parsed by this prototype. "
            "It may be signed, compressed, encrypted, or use a different schema."
        )
    return parsed, None


def _normalize_key(key):
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _normalize_value(value):
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,.;")


def _find_qr_value(qr_fields, aliases):
    normalized = {_normalize_key(k): v for k, v in qr_fields.items()}
    for alias in aliases:
        key = _normalize_key(alias)
        if key in normalized:
            return normalized[key]
    for key, value in normalized.items():
        if any(alias in key for alias in aliases):
            return value
    return None


def compare_ocr_qr(ocr_fields, qr_payload, doc_type):
    qr_fields, parse_error = parse_qr_payload(qr_payload)
    if parse_error:
        return [], qr_fields, parse_error

    comparisons = []
    if doc_type == "Aadhaar Card":
        ocr_aadhaar = ocr_fields.get("Aadhaar number")
        qr_aadhaar = _find_qr_value(
            qr_fields, ["uid", "aadhaar", "aadhaar number", "uid number", "uidai"]
        )
        if ocr_aadhaar and qr_aadhaar:
            ocr_last4 = re.sub(r"\D", "", ocr_aadhaar)[-4:]
            qr_last4 = re.sub(r"\D", "", qr_aadhaar)[-4:]
            comparisons.append(("Aadhaar last 4 digits", ocr_last4, qr_last4,
                                "MATCH" if ocr_last4 == qr_last4 else "MISMATCH"))

        ocr_dob = ocr_fields.get("Date of birth")
        qr_dob = _find_qr_value(qr_fields, ["dob", "date of birth", "dateofbirth"])
        if ocr_dob and qr_dob:
            comparisons.append(("Date of birth", ocr_dob, qr_dob,
                                "MATCH" if _normalize_value(ocr_dob) == _normalize_value(qr_dob) else "REVIEW"))
        ocr_name = ocr_fields.get("Name")
        qr_name = _find_qr_value(qr_fields, ["name", "resident name", "person name"])
        if ocr_name and qr_name:
            same = _normalize_value(ocr_name) == _normalize_value(qr_name)
            comparisons.append(("Name", ocr_name, qr_name, "MATCH" if same else "REVIEW"))

    elif doc_type == "PAN Card":
        ocr_pan = ocr_fields.get("PAN number")
        qr_pan = _find_qr_value(qr_fields, ["pan", "pan number", "permanent account number"])
        if ocr_pan and qr_pan:
            comparisons.append(("PAN number", ocr_pan, qr_pan,
                                "MATCH" if _normalize_value(ocr_pan) == _normalize_value(qr_pan) else "MISMATCH"))
        ocr_dob = ocr_fields.get("Date of birth")
        qr_dob = _find_qr_value(qr_fields, ["dob", "date of birth", "dateofbirth"])
        if ocr_dob and qr_dob:
            comparisons.append(("Date of birth", ocr_dob, qr_dob,
                                "MATCH" if _normalize_value(ocr_dob) == _normalize_value(qr_dob) else "REVIEW"))

    if not comparisons:
        return [], qr_fields, (
            "QR payload was parsed, but no comparable fields were found in both OCR and QR data."
        )
    return comparisons, qr_fields, None


# ---------------------------------------------------------
# Original VerifyAI interface (preserved)
# ---------------------------------------------------------

st.set_page_config(
    page_title="VerifyAI | Document Review",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: #f7f7f5;
        color: #292b2e;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .verify-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.4rem;
    }
    .verify-mark {
        width: 42px;
        height: 42px;
        border-radius: 9px;
        background: #292b2e;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
        font-weight: 600;
    }
    .verify-name {
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.2;
        color: #292b2e;
    }
    .verify-subtitle {
        color: #777973;
        font-size: 0.88rem;
        margin-top: 3px;
    }
    .section-kicker {
        color: #777973;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #deded8;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .muted {
        color: #777973;
        font-size: 0.9rem;
    }
    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #c9c9c1;
        border-radius: 10px;
        padding: 0.8rem;
    }
    div.stButton > button {
        background: #bd704d;
        color: #ffffff;
        border: 1px solid #bd704d;
        border-radius: 7px;
    }
    div.stButton > button:hover {
        background: #a95f40;
        color: #ffffff;
        border-color: #a95f40;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1e1db;
        padding: 0.8rem;
        border-radius: 9px;
    }
    hr { border-color: #e2e2dc; }
    header[data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stToolbar"] { background: transparent !important; }
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1px dashed #bd704d !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #bd704d !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .va-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: #f3e7df;
        color: #9d583b;
        font-size: 21px;
        margin-right: 10px;
        vertical-align: middle;
    }
    .va-icon-title {
        display: flex;
        align-items: center;
        font-size: 20px;
        font-weight: 600;
        color: #292b2e;
        margin-bottom: 12px;
    }
    textarea {
        background: #ffffff !important;
        color: #292b2e !important;
        -webkit-text-fill-color: #292b2e !important;
    }

    /* Strong override for Streamlit status messages (success/info/warning/error) */
    div[data-testid="stAlert"] {
        opacity: 1 !important;
        filter: none !important;
    }
    div[data-testid="stAlert"] > div {
        opacity: 1 !important;
    }
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] *,
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] label {
        color: #242424 !important;
        -webkit-text-fill-color: #242424 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    div[data-testid="stAlert"] svg {
        opacity: 1 !important;
    }

    /* Fix blank metric cards: force dark text on the existing white cards */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        color: #292b2e !important;
    }
    [data-testid="stMetric"] *,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #292b2e !important;
        -webkit-text-fill-color: #292b2e !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] p {
        color: #292b2e !important;
        -webkit-text-fill-color: #292b2e !important;
        opacity: 1 !important;
    }

    /* Keep ordinary captions and muted helper text readable */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    .muted {
        color: #62645f !important;
        -webkit-text-fill-color: #62645f !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="verify-header">
    <div class="verify-mark">V</div>
    <div>
        <div class="verify-name">VerifyAI</div>
        <div class="verify-subtitle">Document Verification</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown('<div class="section-kicker">Document review / 01</div>', unsafe_allow_html=True)
st.title("Review a document")
st.markdown(
    '<p class="muted">Upload a document to inspect image quality, extract text with OCR, and check QR readability.</p>',
    unsafe_allow_html=True
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown("### Upload document")
    st.markdown('<div class="muted">Supported formats: PDF, JPG and PNG</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["jpg", "jpeg", "png", "pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file is None:
        st.markdown("""
        <div class="panel">
            <strong>Before you upload</strong>
            <p class="muted">Use a clear, well-lit image. Make sure the document edges and any QR code are visible.</p>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("### Checks included")
    st.markdown("""
    <div class="panel">
        <strong>Image quality</strong>
        <p class="muted">Basic resolution and sharpness checks.</p>
    </div>
    <div class="panel">
        <strong>QR readability</strong>
        <p class="muted">Attempts to detect and decode supported QR codes.</p>
    </div>
    <div class="muted">These checks do not independently confirm whether a document is genuine.</div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    if "verifyai_seen_hashes" not in st.session_state:
        st.session_state.verifyai_seen_hashes = set()
    duplicate = file_hash in st.session_state.verifyai_seen_hashes
    st.session_state.verifyai_seen_hashes.add(file_hash)

    st.divider()
    st.markdown("### Inspection results")
    st.markdown(f'<div class="muted">File: {uploaded_file.name}</div>', unsafe_allow_html=True)

    all_page_text = []
    quality_issues = []
    ocr_errors = []
    qr_data = None
    page_images = []

    if uploaded_file.type in ["image/jpeg", "image/png"]:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        page_images = [image]
        preview_col, results_col = st.columns([1, 1], gap="large")

        with preview_col:
            st.markdown("#### Document preview")
            st.image(image, use_container_width=True)

        with results_col:
            st.markdown("#### Image quality")
            width, height, sharpness, issues = check_image_quality(image)
            metric1, metric2 = st.columns(2)
            with metric1:
                st.metric("Width", f"{width} px")
            with metric2:
                st.metric("Height", f"{height} px")
            st.metric("Sharpness score", f"{sharpness:.2f}")
            if issues:
                for issue in issues:
                    st.warning(issue)
                    quality_issues.append(issue)
            else:
                st.success("No basic image quality issues detected.")

            st.markdown("#### QR inspection")
            qr_data = detect_qr(image)

            lines, error = show_ocr_results(image)
            if error:
                ocr_errors.append(error)
            if lines:
                all_page_text.append((1, lines))

    elif uploaded_file.type == "application/pdf":
        try:
            pdf = pdfium.PdfDocument(file_bytes)
            if len(pdf) > 0:
                first_page = pdf[0]
                first_image = first_page.render(scale=2).to_pil().convert("RGB")
                page_images = [first_image]
                preview_col, results_col = st.columns([1, 1], gap="large")

                with preview_col:
                    st.markdown("#### PDF preview · Page 1")
                    st.image(first_image, use_container_width=True)

                with results_col:
                    st.markdown("#### QR inspection")
                    qr_data = detect_qr(first_image)

                st.markdown("#### PDF text extraction (all pages)")
                progress = st.progress(0)
                total_pages = len(pdf)
                st.write(f"PDF pages detected: {total_pages}")
                for page_index in range(total_pages):
                    page_image = first_image if page_index == 0 else pdf[page_index].render(scale=2).to_pil().convert("RGB")
                    if page_index > 0:
                        page_images.append(page_image)

                    width, height, sharpness, issues = check_image_quality(page_image)
                    quality_issues.extend([f"Page {page_index + 1}: {x}" for x in issues])

                    with st.spinner(f"Reading page {page_index + 1} of {total_pages}..."):
                        lines, error = extract_text(page_image)

                    if error:
                        ocr_errors.append(f"Page {page_index + 1}: {error}")
                        st.error(f"OCR failed on page {page_index + 1}: {error}")
                    elif lines:
                        all_page_text.append((page_index + 1, lines))
                        st.success(f"Page {page_index + 1}: recognized {len(lines)} text lines.")
                        with st.expander(f"View extracted text · Page {page_index + 1}", expanded=(page_index == 0)):
                            st.text("\n".join(lines))
                    else:
                        st.warning(f"No readable text found on page {page_index + 1}.")
                    progress.progress((page_index + 1) / total_pages)

                st.caption("QR inspection is performed on the first PDF page. OCR is performed on every page.")
            else:
                st.error("The PDF has no pages.")
        except Exception as exc:
            st.error(f"Could not read PDF: {exc}")

    # Shared analysis based on extracted text
    combined_text = "\n\n".join(
        f"PAGE {page_no}\n" + "\n".join(lines)
        for page_no, lines in all_page_text
    )

    if all_page_text:
        st.download_button(
            "Download all extracted text",
            data=combined_text,
            file_name="verifyai_extracted_text.txt",
            mime="text/plain"
        )

    doc_type = classify_document(combined_text) if combined_text else "Unknown / Other"
    fields = extract_fields(combined_text) if combined_text else {}
    checks = validate_fields(combined_text, doc_type) if combined_text else [
        ("OCR extraction", False, "No text available for downstream checks")
    ]

    if combined_text:
        st.divider()
        st.markdown("### Document classification")
        st.write(f"**Detected document type:** {doc_type}")
        st.markdown("#### Extracted fields")
        if fields:
            for key, value in fields.items():
                st.write(f"- **{key}:** {value}")
        else:
            st.info("No supported fields confidently extracted.")

        st.markdown("#### Basic format validation")
        for name, passed, detail in checks:
            if passed:
                st.success(f"{name}: {detail}")
            else:
                st.warning(f"{name}: {detail}")

    # OCR vs QR comparison
    st.divider()
    st.markdown("### OCR vs QR data comparison")
    comparisons = []
    qr_parse_error = None
    qr_fields = {}
    if qr_data and fields:
        comparisons, qr_fields, qr_parse_error = compare_ocr_qr(fields, qr_data, doc_type)
        if comparisons:
            for field_name, ocr_value, qr_value, result in comparisons:
                with st.container(border=True):
                    st.write(f"**{field_name}**")
                    c1, c2, c3 = st.columns([1, 1, 0.8])
                    with c1:
                        st.caption("OCR value")
                        st.write(ocr_value)
                    with c2:
                        st.caption("QR value")
                        st.write(qr_value)
                    with c3:
                        if result == "MATCH":
                            st.success(result)
                        elif result == "MISMATCH":
                            st.error(result)
                        else:
                            st.warning(result)
            st.caption("Comparison is limited to fields successfully extracted and parsed from both sources.")
        else:
            st.info(qr_parse_error or "No comparable fields available.")
    elif not qr_data:
        st.info("Comparison unavailable because no QR payload was decoded.")
    else:
        st.info("Comparison unavailable because OCR did not extract comparable fields.")

    mismatch_count = sum(1 for _, _, _, result in comparisons if result == "MISMATCH")
    review_count = sum(1 for _, _, _, result in comparisons if result == "REVIEW")
    match_count = sum(1 for _, _, _, result in comparisons if result == "MATCH")

    # Screening indicators
    st.divider()
    st.markdown("### Additional screening checks")
    if duplicate:
        st.warning("Exact duplicate file detected in this app session.")
    else:
        st.success("No exact duplicate found in this app session.")
    st.caption("Hash matching detects byte-identical files only, not edited or visually similar copies.")

    ela_score = None
    if page_images:
        try:
            ela_score, ela_image = image_manipulation_indicator(page_images[0])
            st.write(f"**ELA average difference:** {ela_score:.2f}")
            st.image(ela_image, caption="Contrast-enhanced ELA indicator", use_container_width=True)
            st.caption("ELA is a heuristic only. It cannot prove tampering or authenticity.")
        except Exception as exc:
            st.info(f"ELA check unavailable: {exc}")

    # Transparent rule-based screening score; not a probability of fraud.
    risk = 0
    reasons = []
    if quality_issues:
        risk += min(20, 10 * len(quality_issues))
        reasons.extend(quality_issues)
    if ocr_errors:
        risk += 20
        reasons.append("OCR processing error on one or more pages")
    if doc_type == "Unknown / Other":
        risk += 20
        reasons.append("Document type not identified")
    if any(not passed for _, passed, _ in checks):
        risk += 15
        reasons.extend([detail for _, passed, detail in checks if not passed])
    if duplicate:
        risk += 30
        reasons.append("Exact duplicate file detected in this session")
    if mismatch_count:
        risk += min(40, mismatch_count * 20)
        reasons.append(f"{mismatch_count} OCR/QR field mismatch(es)")
    if review_count:
        reasons.append(f"{review_count} OCR/QR field(s) need manual review")
    if qr_data and not comparisons and qr_parse_error:
        reasons.append("QR decoded but comparison could not be completed")
    risk = min(risk, 100)
    outcome = "Manual Review" if risk >= 60 else ("Needs Review" if risk >= 30 else "Low Screening Risk")

    st.divider()
    st.markdown("### Screening result")
    score_col, outcome_col = st.columns(2)
    with score_col:
        st.metric("Screening risk score", f"{risk}/100")
    with outcome_col:
        st.metric("Suggested outcome", outcome)
    if reasons:
        st.markdown("**Reasons / evidence**")
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.success("No configured screening flags were triggered.")

    report = {
        "application": "VerifyAI",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "filename": uploaded_file.name,
        "sha256": file_hash,
        "duplicate_in_current_session": duplicate,
        "document_type": doc_type,
        "page_count": len(page_images),
        "ocr_line_count": sum(len(lines) for _, lines in all_page_text),
        "extracted_fields": fields,
        "validation_checks": [
            {"check": name, "passed": passed, "detail": detail}
            for name, passed, detail in checks
        ],
        "qr_decoded": bool(qr_data),
        "qr_payload_parsed": bool(qr_fields),
        "qr_comparison": [
            {"field": name, "ocr_value": ocr_value, "qr_value": qr_value, "result": result}
            for name, ocr_value, qr_value, result in comparisons
        ],
        "qr_comparison_note": qr_parse_error,
        "comparison_summary": {
            "matches": match_count,
            "mismatches": mismatch_count,
            "manual_review": review_count
        },
        "quality_issues": quality_issues,
        "ocr_errors": ocr_errors,
        "ela_average_difference": ela_score,
        "screening_risk_score": risk,
        "suggested_outcome": outcome,
        "reasons": reasons,
        "limitations": [
            "Prototype screening only; not official document authentication.",
            "OCR can make mistakes; verify extracted values against the source.",
            "QR decoding does not prove authenticity.",
            "QR parsing supports common JSON, XML and key-value formats only.",
            "Some Aadhaar QR payloads may require specialized decoding and signature validation.",
            "Only fields available in both OCR and parsed QR payload are compared.",
            "File hash detects exact copies only.",
            "ELA is heuristic and cannot prove tampering.",
            "Risk score is a rule-based screening score, not a fraud probability.",
            "No issuing-authority database verification is connected."
        ]
    }
    st.download_button(
        "Download verification report (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="verifyai_verification_report.json",
        mime="application/json"
    )

    st.warning(
        "VerifyAI is a prototype screening tool. It does not connect to UIDAI, "
        "Income Tax, passport or other issuing-authority databases. Do not treat "
        "its result as proof that a document is genuine or fraudulent."
    )
