import sys
import streamlit as st
import awesome_streamlit as ast

sys.path.append('src')
from generate_classification_config import generate_config
from dolphin.app.functions import homepage, classify, detect, raven_classify, visualize_augmentation

FUNCTIONALITIES = {
    "Home Page": homepage,
    "Classification": classify,
    "Detection" : detect,
    "Classify Prior Detections": raven_classify,
    "Visualize Augmentations": visualize_augmentation
}

def main():
    
    generate_config()

    st.set_page_config(
     page_title="Dolphin Whistles",
     page_icon="🐬",
     initial_sidebar_state="expanded",
    )

    st.title("Dolphin Whistles")

    st.warning('If you want to rerun any of the pages, you have to REFRESH the page.')

    fn = st.radio("What do you want to do?", list(FUNCTIONALITIES.keys()))
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    if fn:
        FUNCTIONALITIES[fn].main()
    else:
        st.write("Please make a selection")

    st.write("© [Allen Institute for AI](https://allenai.org/) All Rights Reserved | [Privacy Policy](https://allenai.org/privacy-policy) | [Terms of Use](https://allenai.org/terms) | [Business Code of Conduct](https://allenai.org/business-code-of-conduct)")

if __name__ == '__main__':
    main()
