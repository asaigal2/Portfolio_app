import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
import openai
from langchain.llms import OpenAI  # Corrected import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.llms import OpenAI 
import os
OpenAI_key = os.environ.get("OPENAI_API_KEY")

if OpenAI_key is None:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
else:
    openai.api_key = OpenAI_key



def first_page():
    st.title('Hi I am Ayushi! 👋')
    st.write('I am a senior at the University of Wisconsin-Madison studying Electrical Engineering with a specialization in Data Science and Machine Learning. I am also pursuing a minor in Computer Sciences. I am so happy you are here!')
    url = 'Resume_Ayushi_Saigal.pdf'

    st.markdown(f'''
    <a href={url}><button style="background-color:Pink;">Check out my Resume</button></a>
    ''', unsafe_allow_html=True)
    st.header("Meet AyushiBot!")
    st.write("Hello! I am AyushiBot. Ask me questions about Ayushi! ")

    # Define AI agent functionality
    def ask_bot(input_text):
        # Define LLM
        
            llm = OpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=OpenAI_key,
            )
            service_context = ServiceContext.from_defaults(llm=llm)
            
            # Load the file
            documents = SimpleDirectoryReader(input_files=["bio.txt"]).load_data()

            # Load index
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)    

            # Query LlamaIndex and GPT-3.5 for the AI's response
            PROMPT_QUESTION = """You are an AI agent named AyushiBot helping answer questions about Ayushi to recruiters. Introduce yourself when you are introducing who you are.
            If you do not know the answer, politely admit it and let users know how to contact Ayushi to get more information. 
            Human: {input}
            """
            output = index.as_query_engine().query(PROMPT_QUESTION.format(input=input_text))
            print(f"output: {output}")
            return output.response
            
        

    def get_text():
        input_text = st.text_input("You can send your questions and hit Enter to know more about me from my AI agent, AyushiBot!", key="input")
        return input_text

    user_input = get_text()

    if user_input:
        st.info(ask_bot(user_input))


def second_page():
    st.markdown("<h1 style='margin-bottom: 20px;'>Research Experience:</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        image_path = 'Presentation_Image_Processing.jpeg'
        st.image(image_path, caption="Poster Presentation")
        
        st.markdown("""
        **Image Processing workflow used**  
        1. **Fourier Filtering**  
        2. **Baseline Calculation**  
        3. **Thresholding**  
        4. **Erosion and Dilation**  
        5. **Blob Detection**  
        6. **Visualize Centroids**
        """)

    with col2:
        st.markdown("**Project 1: Summer 2023-Image Processing Intern at UW Hospital**")
        st.markdown('''
        I had the incredible opportunity to work individually at the Medical Physics X-Ray Imaging (MOXI) lab, where I focused on  quantifying the levels of endogenous Protoporphyrin-IX (PPIX) in the pores of the skin. This work is critical for enhancing the effectiveness of photodynamic therapy (PDT), a treatment modality for various skin conditions and cancers.
        During this project, I employed **MATLAB extensively for data processing and analysis.** MATLABs robust capabilities in image processing allowed me to apply various filtering techniques to enhance and isolate relevant features in the skin images.
        One of the key methodologies I utilized was **machine learning, specifically k-means clustering**. 
        By clustering pixels based on their intensity and color features, I could distinguish areas with high concentrations of PPIX from those with lower levels.
        To ensure the reliability and **statistical significance of my findings, I conducted t-tests**.
        These statistical tests helped compare the PPIX levels before and after the application of treatments, determining the efficacy of the interventions. 
        ''')

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        image_path = 'MathResearch.PNG'
        st.image(image_path, caption="As seen on the lab's website")

    with col2:
        st.markdown("                                                               ")
        st.markdown("**Project 2: Math Researcher at Madison Experimental Mathematics Lab**")
        st.markdown('''At the Madison Experimental Mathematics Lab, I delved into the complexities of three-dimensional slow k-nim games. 
                    These **combinatorial games** involve players taking turns to remove stones from heaps based on specific rules, with the goal of being the one to remove the last stone.
                     My research focused on identifying optimal strategies by categorizing game positions as either P-positions, where the previous player can force a win, or N-positions, where the next player has a winning strategy.
                     Through **recursive analysis and strategic evaluation**, I explored how players can navigate these complex configurations to force their opponents into disadvantageous positions, ultimately leading to a win. This experience enhanced my understanding of **advanced mathematical concepts** and their applications in game theory, 
                    building on principles from one and two-dimensional nim games and adapting them to a more intricate three-dimensional context.
        ''')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Project 3: Machine Learning Researcher for Medical Imaging Denoising**")
    st.markdown('''
    During my independent study, I focused on leveraging **neural network technologies** to enhance medical imaging, particularly in challenging environments where noise-free data is scarce. My research project, inspired by the "Noise2Noise" framework from Nvidia, aimed to develop a neural network denoiser capable of training with noisy datasets typical of clinical settings. Specifically, I applied this methodology to Fluorescence Guided Surgery (FGS), a crucial technique in modern surgical procedures that relies on fluorescent markers to distinguish between healthy and diseased tissues. The inherent noise in FGS imaging—stemming from shot noise, camera sensor imperfections, and laser leakage light—compromises image clarity and surgical precision. To address these challenges, 
                I created custom datasets augmented with synthetic noise, allowing the neural network to learn denoising under realistic conditions. The training loop integrated these noise-augmented frames, enabling the model to predict clean states from noisy inputs. This approach not only aimed to enhance practical deployment in healthcare but also contributed to the broader understanding of **noise modeling and machine learning theory**. 
                The project's success in improving image clarity has significant implications for better surgical outcomes and the adoption of advanced imaging techniques.
    ''')

st.sidebar.title("Navigation")
st.markdown(f'''
    <style>
    section[data-testid="st.sidebar"] .css-ng1t4o {{width: 14rem;}}
    </style>
''', unsafe_allow_html=True)
page = st.sidebar.selectbox("Go to", ["About me!","Research", "Personal Projects"])

# Display the selected page
if page == "About me!":
    first_page()
elif page == "Research":
    second_page()
elif page == "Personal Projects":
    st.write('Under construction...')
