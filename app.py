import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="AWARE",
                   layout="wide",
                   page_icon="🧑‍⚕️")

working_dir = os.path.dirname(os.path.abspath(__file__))

autism_model = pickle.load(open(f'{working_dir}/autism_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu('AWARE',

                           ['Questionnaire Analysis',
                            'Eye Tracking',],
                           menu_icon='hospital',
                           icons=['clipboard-data','eye'],
                           default_index=0)


if selected == 'Questionnaire Analysis':
    st.title('Questionaire Analysis using ML')

    

if selected == 'Eye Tracking':
    st.title('Eye Tracing using Computer Vision')