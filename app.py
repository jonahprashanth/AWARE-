import os
import time
import pickle
import cv2
import pandas as pd
import streamlit as st
import tensorflow as tf
import numpy as np
import altair as alt
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="AWARE",
                   layout="wide",
                   page_icon="👨🏻‍⚕️")

# Define working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the autism prediction model
try:
    autism_model = pickle.load(open(os.path.join(working_dir, 'Questionnaire_Analysis/autism_model.sav'), 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the Computer Vision model
try:
    computer_vision_model = tf.keras.models.load_model(os.path.join(working_dir, 'Computer_Vision/saved_models/resnet50_autism_classifier.h5'))
except Exception as e:
    st.error(f"Error loading computer vision model: {e}")
    st.stop()

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="AWARE",
        options=['Questionnaire Analysis', 'Computer Vision', 'Result'],
        icons=['list-task', 'eye'],  
        menu_icon="hospital",
        default_index=0,  # default selected menu item
    )

# Function to preprocess and predict using the computer vision model
def predict_computer_vision(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Assuming the model expects 224x224 input size
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    prediction = computer_vision_model.predict(img)
    confidence = prediction[0][0]
    return confidence

if 'qr' not in st.session_state:
    st.session_state.qr = 0
if 'cr' not in st.session_state:
    st.session_state.cr = 0
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""

# Questionnaire Analysis section
if selected == 'Questionnaire Analysis':
    st.title('Questionnaire Analysis')

    name = st.text_input("Name of the Patient")
    if name:
        st.session_state.patient_name = name
    
    c1, c2 = st.columns(2)
    yesorno = {"Yes": 1, "No": 0}

    with c1:
        sex_mapping = {"Male": 1, "Female": 0}
        sex = sex_mapping[st.radio("Gender", ["Male", "Female"])]
    with c2:
        ethnicity_mapping = {
            "Hispanic": 0, "Latino": 1, "Native Indian": 2, "Others": 3, "Pacific": 4,
            "White European": 5, "Asian": 6, "Black": 7, "Middle Eastern": 8, "Mixed": 9, "South Asian": 10
        }
        ethnicity = ethnicity_mapping[st.selectbox("Ethnicity", list(ethnicity_mapping.keys()))]
    with c1:
        jaundice = yesorno[st.radio("Presence of Jaundice", ["Yes", "No"])]
    with c2:
        family_mem_with_ASD = yesorno[st.radio("Family member with ASD", ["Yes", "No"])]
    with c1:
        a1 = yesorno[st.radio("Does your child look at you when you call his/her name?", ["Yes", "No"])]
    with c2:
        a2 = yesorno[st.radio("How easy is it for you to get eye contact with your child?", ["Yes", "No"])]
    with c1:
        a3 = yesorno[st.radio("Does your child point to indicate that he/she wants something?", ["Yes", "No"])]
    with c2:
        a4 = yesorno[st.radio("Does your child point to share interest with you?", ["Yes", "No"])]
    with c1:
        a5 = yesorno[st.radio("Does your child pretend?", ["Yes", "No"])]
    with c2:
        a6 = yesorno[st.radio("Does your child follow where you’re looking?", ["Yes", "No"])]

    with c1:
        a7 = yesorno[st.radio("If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?", ["Yes", "No"])]
    with c2:
        a8 = yesorno[st.radio("Were your child's first words clear and understandable?", ["Yes", "No"])]
    with c1:
        a9 = yesorno[st.radio("Does your child use simple gestures?", ["Yes", "No"])]
    with c2:
        a10 = yesorno[st.radio("Does your child stare at nothing with no apparent purpose?", ["Yes", "No"])]

    if st.button('Autism Test Result'):
        inputs = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, sex, ethnicity, jaundice, family_mem_with_ASD]
        inputs = [int(x) for x in inputs]

        try:
            autism_prediction = autism_model.predict([inputs])
            autism_confidence = autism_model.predict_proba([inputs])
            result = 'Autistic' if autism_prediction[0] == 0 else 'Not Autistic'
            confidence = autism_confidence[0][autism_prediction[0]] * 100
            
            st.session_state.qr = autism_confidence[0][autism_prediction[0]]
            
            st.subheader(f'{result}')
            st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

            if name:
                st.subheader(f'Results for {name}')
            else:
                st.subheader('Results')
            
            data = {
                'Questions': [
                    "Look at you when called", "Eye contact ease", "Point for wants", "Point to share interest",
                    "Pretend", "Follow where looking", "Comfort when upset", "First words", "Use simple gestures", "Stare at nothing"
                ],
                'Responses': inputs[:10]
            }
            response_mapping = {1: "Yes", 0: "No"}
            data['Responses'] = [response_mapping[response] for response in data['Responses']]
            df = pd.DataFrame(data)
            st.table(df)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Computer Vision
if selected == 'Computer Vision':
    st.title('Computer Vision')
    st.write("Instructions:")
    st.write("- Read the paragraph after starting the test.")
    st.write("- Click the 'Start Test' button to begin.")
    
    # Define a directory to save images
    img_dir = os.path.join(working_dir, 'captured_images')
    os.makedirs(img_dir, exist_ok=True)

    if st.button('Start Test'):
        st.markdown("""
        ## Read the paragraph
        Once, amidst the verdant pastures of Vrindavan, where the sweet fragrance of blooming flowers mingled
        with the melodious songs of birds, lived Lord Krishna, the epitome of divine love and wisdom. His 
        enchanting flute melodies echoed through the lush groves, captivating the hearts of both mortals and 
        celestial beings alike. The young cowherd, with his playful demeanor and profound teachings, drew 
        devotees from far and wide, each seeking solace in his divine presence. Legends spoke of his childhood
        antics, his endearing bond with cows, and his valorous deeds against the forces of darkness. 
        """)
        
        # Capture images for 10 seconds
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        img_count = 0
        results = []

        while img_count < 10:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            img_count += 1
            img_path = os.path.join(img_dir, f'image_{img_count}.jpg')
            cv2.imwrite(img_path, frame)

            # Validate the image using the classifier
            confidence = predict_computer_vision(frame)
            results.append(confidence)

            time.sleep(1)

        cap.release()
        st.success(f'Captured {img_count} images.')

        average_prediction = np.mean(results)
        st.session_state.cr = average_prediction
        threshold = 0.5
        final_result = 'Autistic' if average_prediction < threshold else 'Not Autistic'
        
        st.markdown(f"## Computer Vision Test Completed")
        st.subheader(f"Prediction : **{final_result}**")
        
        if final_result == 'Autistic':
            temp1 = 100 - average_prediction * 100
            st.metric(label="Model Confidence", value=f"{temp1:.2f}%")
        else:
            st.metric(label="Model Confidence", value=f"{average_prediction * 100:.2f}%")

        # Displaying images captured
        st.markdown("## Captured Images")
        st.write("Here are the images captured during the test:")

        image_files = sorted(os.listdir(img_dir))

        num_images = len(image_files)
        num_columns = 5
        num_rows = (num_images // num_columns) + (1 if num_images % num_columns != 0 else 0)

        for i in range(num_rows):
            columns = st.columns(num_columns)
            for j in range(num_columns):
                index = i * num_columns + j
                if index < num_images:
                    image_path = os.path.join(img_dir, image_files[index])
                    columns[j].image(image_path, use_column_width=True, caption=f"Image {index + 1}")

        # Display confidence levels
        st.markdown("## Computer Vision Confidence Levels")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("Image Index")
            for i in range(1, img_count + 1):
                st.write(i)
        with c2:
            st.write("Autistic (%)")
            for conf in results:
                temp = 100 - conf * 100
                st.write(f"{temp:.2f}%")
        with c3:
            st.write("Non-Autistic (%)")
            for conf in results:
                st.write(f"{conf * 100:.2f}%")

        # Plot the results
        st.markdown("## Computer Vision Test Results")

        # Plotting using Altair
        df_results = pd.DataFrame({
            'Image Index': range(1, img_count + 1),
            'Prediction': results
        })
        line_chart = alt.Chart(df_results).mark_line(point=True).encode(
            x='Image Index',
            y='Prediction'
        ).properties(
            title='Computer Vision Predictions'
        )
        threshold_line = alt.Chart(pd.DataFrame({'y': [threshold]})).mark_rule(color='red').encode(y='y')
        st.altair_chart(line_chart + threshold_line, use_container_width=True)

        st.write("Thank you for participating in the computer vision test.")

# Result section
if selected == 'Result':
    if st.session_state.patient_name:
        st.title(f'Results of {st.session_state.patient_name}')
    else:
        st.title('Results')
    try:
        # Ensure the session states have been set
        if 'qr' not in st.session_state or 'cr' not in st.session_state:
            st.error("Questionnaire result or Computer Vision result not available.")
        else:
            # Log-odds conversion
            qr_log_odds = np.log(st.session_state.qr / (1 - st.session_state.qr))
            cr_log_odds = np.log(st.session_state.cr / (1 - st.session_state.cr))

            # Combine the log odds
            combined_log_odds = (qr_log_odds + cr_log_odds) / 2

            # Convert back to probability
            combined_probability = 1 / (1 + np.exp(-combined_log_odds))

            result = 'Autistic' if combined_probability < 0.5 else 'Not Autistic'
            confidence = combined_probability * 100 if combined_probability >= 0.5 else (1 - combined_probability) * 100

            st.subheader(f'{result}')
            st.metric(label="Confidence", value=f"{confidence:.2f}%")

    except Exception as e:
        st.error(f"Error during final prediction: {e}")

