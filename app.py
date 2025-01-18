from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline  # Hugging Face's Transformers library for summarization

def main():
    load_dotenv()
    st.set_page_config(page_title="Summarize your PDF")
    st.header("Summarize your PDF 💬")
    
    # Load summarization model
    summarizer = pipeline("summarization")

    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
       # st.write("Full Text Extracted:")
       # st.write(text)
        
        # Summarize the text
        if text.strip():
            st.subheader("Summary:")
            # Split text into chunks if it's too long
            max_chunk_size = 1000  # Adjust based on model limits
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            summary = ""
            for chunk in chunks:
                summarized_chunk = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summary += summarized_chunk[0]['summary_text'] + " "
            
            st.write(summary)
        else:
            st.warning("No text could be extracted from the PDF. Please check the file.")

if __name__ == '__main__':
    main()
