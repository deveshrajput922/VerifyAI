def show_ocr_results(pil_image):
    st.markdown("#### Text extraction (OCR)")

    with st.spinner("Reading text from document..."):
        text_lines, error = extract_text(pil_image)

    if error:
        st.error(f"OCR Error: {error}")
        return

    if not text_lines:
        st.warning("No text detected.")
        return

    extracted_text = "\n".join(text_lines)

    st.success(
        f"Successfully extracted {len(text_lines)} lines."
    )

    st.markdown("### Extracted Document Text")

    st.code(extracted_text, language=None)

    st.download_button(
        "Download Extracted Text",
        data=extracted_text,
        file_name="extracted_text.txt",
        mime="text/plain"
    )