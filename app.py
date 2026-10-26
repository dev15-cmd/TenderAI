import streamlit as st
import qrcode
from PIL import Image
import numpy as np
import hashlib
import json
import base64
import io
import pandas as pd
from zipfile import ZipFile
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Dict, Tuple
import logging

try:
    import fitz  # PyMuPDF for PDF rendering
    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Encryption ----------------
def _derive_key_from_password(password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
    if salt is None:
        salt = hashlib.sha256(password.encode()).digest()[:16]
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

def encrypt_qr_data(data: str, password: str) -> str:
    key, _ = _derive_key_from_password(password)
    f = Fernet(key)
    encrypted = f.encrypt(data.encode())
    return base64.urlsafe_b64encode(encrypted).decode()

def decrypt_qr_data(encrypted_data: str, password: str) -> str:
    try:
        key, _ = _derive_key_from_password(password)
        f = Fernet(key)
        decrypted = f.decrypt(base64.urlsafe_b64decode(encrypted_data.encode()))
        return decrypted.decode()
    except Exception:
        return None

# ---------------- Watermark ----------------
def generate_watermark(market_id: str) -> np.ndarray:
    hash_val = hashlib.sha256(market_id.encode()).digest()
    repeated = (hash_val * (256 // len(hash_val) + 1))[:256]
    return np.frombuffer(repeated, dtype=np.uint8) % 2

def embed_watermark(qr_image: Image.Image, market_id: str) -> Image.Image:
    img_array = np.array(qr_image.convert('RGB')).astype(np.uint8)
    watermark_pattern = generate_watermark(market_id)
    height, width = img_array.shape[:2]
    for i in range(min(len(watermark_pattern), height * width)):
        row, col = i // width, i % width
        val = int(img_array[row, col, 0])
        img_array[row, col, 0] = val | 1 if watermark_pattern[i] else val & 0xFE
    return Image.fromarray(img_array.astype('uint8'))

def verify_watermark(qr_image: Image.Image, expected_market_id: str, threshold: float = 0.85) -> bool:
    img_array = np.array(qr_image.convert('RGB')).astype(np.uint8)
    extracted = [(int(img_array[i // img_array.shape[1], i % img_array.shape[1], 0]) & 1) for i in range(256)]
    expected = generate_watermark(expected_market_id)
    similarity = np.sum(np.array(extracted) == expected[:len(extracted)]) / len(extracted)
    return similarity >= threshold

# ---------------- QR Generation ----------------
def generate_encrypted_qr(encrypted_data: str, market_id: str) -> Image.Image:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(encrypted_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return embed_watermark(qr_img, market_id)

def image_to_bytes(img: Image.Image, format: str = 'PNG') -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

# ---------------- PDF Renderer ----------------
def _render_pdf_first_page_to_image(uploaded_file) -> Image.Image:
    if not _HAS_FITZ:
        raise RuntimeError("Install PyMuPDF with `pip install PyMuPDF`.")
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc.load_page(0)
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

# ---------------- Streamlit App ----------------
def main():
    st.set_page_config(page_title="AI Secure QR Generator", page_icon="🔐", layout="wide")
    st.title("🔐 AI Secure QR Code Generator")

    tab1, tab2 = st.tabs(["📦 Generate QR", "✅ Verify QR"])

    # ---------- TAB 1: Generate QR ----------
    with tab1:
        st.subheader("Generate Secure QR Code")

        option = st.radio("Choose Input Method:", ["Manual Entry", "Upload CSV"])
        doc_type = st.selectbox("📄 Document Type", ["Certificate", "Diploma", "Cheque", "Receipt", "Ticket", "ID Card"])
        password = st.text_input("🔑 Encryption Password", type="password")
        market_id = st.text_input("🏢 Organization/Market ID", value="ORG-2025-001")

        if option == "Manual Entry":
            data_input = st.text_area("Enter JSON Data", value=json.dumps({
                "type": doc_type,
                "student_name": "John Doe",
                "roll_no": "CSE1234",
                "course": "Computer Engineering",
                "year": "2025",
                "result": "Pass"
            }, indent=2))

            if st.button("🚀 Generate QR"):
                try:
                    encrypted = encrypt_qr_data(data_input, password)
                    qr_img = generate_encrypted_qr(encrypted, market_id)
                    st.image(qr_img, caption="Secure QR Code", use_container_width=True)
                    st.download_button("⬇️ Download QR", data=image_to_bytes(qr_img),
                                       file_name=f"{doc_type}_QR.png", mime="image/png")
                except Exception as e:
                    st.error(f"Error: {e}")

        else:  # CSV upload
            uploaded_csv = st.file_uploader("Upload CSV (columns: name, roll_no, course, year, result)", type="csv")
            if uploaded_csv is not None:
                df = pd.read_csv(uploaded_csv)
                st.dataframe(df)
                if st.button("📄 Generate QRs for All Records"):
                    zip_buf = io.BytesIO()
                    with ZipFile(zip_buf, "w") as zipf:
                        for _, row in df.iterrows():
                            record = row.to_dict()
                            record["type"] = doc_type
                            data_json = json.dumps(record)
                            encrypted = encrypt_qr_data(data_json, password)
                            qr_img = generate_encrypted_qr(encrypted, market_id)
                            img_bytes = image_to_bytes(qr_img)
                            zipf.writestr(f"{record.get('roll_no', 'record')}.png", img_bytes)
                    zip_buf.seek(0)
                    st.success("✅ All QRs generated successfully!")
                    st.download_button("⬇️ Download All QRs (ZIP)", data=zip_buf,
                                       file_name=f"{doc_type}_batch_qr_codes.zip", mime="application/zip")

    # ---------- TAB 2: Verify QR ----------
    with tab2:
        st.subheader("Verify QR Code Authenticity")

        uploaded_qr = st.file_uploader("Upload QR Image or PDF", type=["png", "jpg", "jpeg", "pdf"])
        verify_market_id = st.text_input("Expected Market ID", value=market_id)
        verify_password = st.text_input("Decryption Password", type="password", key="verify_pass")

        if st.button("🔍 Verify QR Code") and uploaded_qr:
            try:
                qr_img = _render_pdf_first_page_to_image(uploaded_qr) if uploaded_qr.type == "application/pdf" else Image.open(uploaded_qr).convert("RGB")

                # Step 1: Watermark verification
                with st.spinner("Verifying watermark..."):
                    authentic = verify_watermark(qr_img, verify_market_id)
                    st.success("✅ Authentic QR (Watermark Verified)") if authentic else st.error("❌ Tampered QR Detected")

                # Step 2: Decode and decrypt
                if verify_password:
                    with st.spinner("Decrypting data..."):
                        try:
                            import pyzbar.pyzbar as pyzbar
                            gray = np.array(qr_img.convert("L"))
                            decoded = pyzbar.decode(gray)
                            if decoded:
                                encrypted_data = decoded[0].data.decode()
                                decrypted = decrypt_qr_data(encrypted_data, verify_password)
                                if decrypted:
                                    st.success("✅ Data decrypted successfully!")
                                    st.json(json.loads(decrypted))
                                else:
                                    st.error("❌ Wrong password or invalid QR")
                            else:
                                st.warning("⚠️ Could not decode QR. Try a clearer image.")
                        except Exception as e:
                            st.error(f"Decryption error: {e}")
            except Exception as e:
                st.error(f"Verification error: {e}")

if __name__ == "__main__":
    main()
