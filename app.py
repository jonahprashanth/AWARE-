import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="AWARE",
                   layout="wide",
                   page_icon="👨🏻‍⚕️")

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
    st.title('Questionaire Analysis')
    
    name = st.text_input("Name of the Pateint")
    c1,c2=st.columns(2)
    yesorno = {"Yes": 1, "No": 0}
    with c1:
        sex_mapping = {"Male": 1, "Female": 0}
        sex = sex_mapping[st.radio("Sex", ["Male", "Female"])]
    with c2:
        ethnicity_mapping = {"Hispanic": 0, "Latino": 1, "Native Indian": 2, "Others": 3, "Pacific": 4, "White European": 5, "Asian": 6, "Black": 7, "Middle Eastern": 8, "Mixed": 9, "South Asian": 10}
        ethnicity = ethnicity_mapping[st.selectbox("Ethnicity", ["Hispanic", "Latino", "Native Indian", "Others", "Pacific", "White European", "Asian", "Black", "Middle Eastern", "Mixed", "South Asian"])]
    with c1:
        jaundice = yesorno[st.radio("Jaundice", ["Yes", "No"])]
    with c2:
        family_mem_with_ASD = yesorno[st.radio("Family member with ASD", ["Yes", "No"])]
    with c1:
        a1 = yesorno[st.radio("Does your child look at you when you call his/her name?", ["Yes", "No"])]
    with c2:
        a2 = yesorno[st.radio("How easy is it for you to get eye contact with your child?", ["Yes", "No"])]
    with c1:
        a3 = yesorno[st.radio("Does your child point to indicate that s/he wants something?", ["Yes", "No"])]
    with c2:
        a4 = yesorno[st.radio("Does your child point to share interest with you?", ["Yes", "No"])]
    with c1:
        a5 = yesorno[st.radio("Does your child pretend?", ["Yes", "No"])]
    with c2:
        a6 = yesorno[st.radio("Does your child follow where you’re looking?", ["Yes", "No"])]

    with c1:
        a7 = yesorno[st.radio("If you or someone else in the family is visibly upset, does your child show signs of wan9ng to comfort them?", ["Yes", "No"])]
    with c2:
        a8 = yesorno[st.radio("Would you describe your child’s first words as:", ["Yes", "No"])]
    with c1:
        a9 = yesorno[st.radio("Does your child use simple gestures?", ["Yes", "No"])]

    with c2:
        a10 = yesorno[st.radio("Does your child stare at nothing with no apparent purpose?", ["Yes", "No"])]

    result = ''
    
    if st.button('Autism Test Result'):
        inputs=[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,sex,ethnicity,jaundice,family_mem_with_ASD]
        inputs = [int(x) for x in inputs]
        autism_prediction=autism_model.predict([inputs])
        if autism_prediction[0]==1:
            if name=='':
                st.success('The Person is Autistic')
            else:
                st.success(f'{name} is Autistic')
        else:
            if name=='':
                st.success('The Person is not Autistic')
            else:
                st.success(f'{name} is not Autistic')

if selected == 'Eye Tracking':
    st.title('Eye Tracing using Computer Vision')