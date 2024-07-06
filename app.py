import streamlit as st

import core

# Page title
st.set_page_config(page_title='Docktor AI_bolit', page_icon='💊')
st.title('🩺 **Docktor** :red[AI]**bolit**')

uploaded_file = st.file_uploader('Upload a file', type='pdf')
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

if query_text:
    with st.spinner('Calculating...'):
        res = core.generate(uploaded_file, query_text)
        st.write(res)
