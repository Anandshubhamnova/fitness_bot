import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Fitness Plan Generator", page_icon="🏋️")
st.title("🏋️ Fitness Plan Generator")

question = st.text_input("Enter your question:", placeholder="e.g., Give me a beginner cardio plan")

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Fetching your answer..."):
            try:
                response = requests.get(API_URL, params={"q": question}, stream=True)
                answer = ""
                # Stream response chunks
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        answer += chunk.decode("utf-8")
                        st.write(answer)  # Update progressively
                st.success("Answer fetched successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")