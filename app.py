import streamlit as st

st.set_page_config(
    page_title="VerifyAI",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 VerifyAI")
st.subheader("AI-Powered Document Fraud Detection")

st.write(
    "Upload a document to screen it for "
    "potential inconsistencies."
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    st.success("Document uploaded successfully!")

    st.image(
        uploaded_file,
        caption="Uploaded Document",
        use_container_width=True
    )

    if st.button("Analyze Document"):

        st.info(
            "Upload successful. Document analysis "
            "will be implemented in the next stage."
        )