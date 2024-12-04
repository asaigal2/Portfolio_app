import os
import openai
import streamlit as st
import base64
import openai
import os
import streamlit as st
import base64
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title='Portfolio',  # Title of the browser tab
    layout="wide",           # Use the wide layout
    page_icon='👧🏻'
)

# Set OpenAI API key
api_key = st.secrets["OPENAI_API_KEY"]

if not openai.api_key:
    st.error("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize session state to track the current section
if 'section' not in st.session_state:
    st.session_state.section = 'About Me'  # Default section is 'About Me'

# Function to display PDF
def get_pdf_display(pdf_file):
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="600" type="application/pdf"></iframe>'
    return pdf_display

# Function to render the "About Me" section
def render_about_me():
    st.title('Hi, I am Ayushi! 👋')
    st.write('I am a senior at the University of Wisconsin-Madison studying Electrical Engineering with a specialization in Data Science and Machine Learning. I am also pursuing a minor in Computer Sciences. I am so happy you are here!')

    # Display Resume Section
    resume_file = 'Saigal_Ayushi_Resume.pdf'

    if 'show_resume' not in st.session_state:
        st.session_state.show_resume = False

    if st.button('Check out my Resume'):
        st.session_state.show_resume = not st.session_state.show_resume

    if st.session_state.show_resume:
        pdf_display = get_pdf_display(resume_file)
        st.markdown(pdf_display, unsafe_allow_html=True)

    # AyushiBot Section
    st.header("Meet AyushiBot!")
    st.write("Hello! I am AyushiBot.")

    # Ensure bio.txt file exists
    try:
        with open("bio.txt", "r") as file:
            bio_content = file.read()
    except FileNotFoundError:
        st.error("bio.txt file not found.")
        return

    # Function to handle OpenAI ChatCompletion
    def ask_bot(input_text):
        try:
        # Initialize client with the API key from secrets
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an AI agent named AyushiBot helping answer questions about Ayushi to recruiters. Here is some information about Ayushi: {bio_content}. If you do not know the answer, politely admit it and let users know to contact Ayushi for more information."
                },
                {"role": "user", "content": input_text}
            ]
        )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error connecting to OpenAI: {str(e)}")
            return "I'm having trouble connecting right now. Please try again later."
    # User Input and Bot Response
    user_input = st.text_input("Ask me anything about Ayushi! (Hobbies, Visa Status, etc.)")
    if user_input:
        bot_response = ask_bot(user_input)
        st.write(bot_response)





# Function to render research experience
def render_research_experience():
    st.markdown("<h1 style='margin-bottom: 20px;'>Technical Experience</h1>", unsafe_allow_html=True)

    # Project 1: ML Science Intern
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display an image related to the technical experience
        st.image('Tech.png', caption="Bennett Coleman - Data Science & Analytics Intern")

    with col2:
        st.markdown("**Data Science and Analytics Intern - Bennett Coleman**")
        st.write("""
            - **Role**: Contributed to the **Business Intelligence** team by delivering data-driven insights and improving data processes.
            - **Python Automation**: Developed Python scripts to automate data collection and preprocessing for datasets exceeding 500,000 rows, reducing manual data entry efforts by **75%**.
            - **SQL & Excel**: Created advanced **SQL queries** and used **Excel functions** (e.g., pivot tables, VLOOKUP) to extract and analyze data from multiple databases, uncovering trends that improved decision-making accuracy.
            - **Data Visualization**: Designed interactive **PowerBI dashboards**, integrating data extracted via **Selenium** from web sources to provide actionable insights.
            - **Elections Data Analysis**: Developed a comprehensive data analysis system for election data, using **PostgreSQL** for data management, and created interactive maps and visualizations to effectively present election results.
            - **Containerized Visualizations**: Utilized **Streamlit** and **Django** to containerize and deploy interactive visualizations, enabling seamless integration of data analytics and dashboards into web applications.
        """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('image.png', caption="Reference picture")
    with col2:
        st.markdown("**Machine Learning Researcher for Medical Imaging Denoising**")
        st.write("""
            - **Objective**: Developed a neural network-based solution to denoise medical images, improving the clarity of images used in **Fluorescence Guided Surgery (FGS)**.
            - **Machine Learning Framework**: Implemented the **Noise2Noise** methodology to train neural networks on noisy data, allowing for noise-free medical images without requiring clean datasets.
            - **Data Augmentation**: Created custom datasets with synthetic noise to simulate real-world conditions, ensuring robust model performance.
            - **Outcome**: The neural network significantly improved the quality of FGS images, enhancing the precision and reliability of surgical outcomes.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project 2: Math Researcher
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('MathResearch.PNG', caption="As seen on the lab's website")
    with col2:
        st.markdown("**Math Researcher at Madison Experimental Mathematics Lab**")
        st.write("""
            - **Objective**: Developed a strategic framework to optimize outcomes in three-dimensional slow k-nim combinatorial games using mathematical modeling and data analysis.
            - **Data Analysis**: Conducted recursive analysis to identify **P-positions** and **N-positions**, using **game theory** and advanced combinatorial optimization techniques.
            - **Quantitative Methods**: Utilized recursive algorithms to simulate game scenarios, optimizing strategies to force opponents into unfavorable positions.
            - **Outcome**: Developed a novel algorithm for analyzing multi-dimensional game structures, improving the understanding of game complexity in combinatorial mathematics.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project 3: Machine Learning Researcher
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('Presentation_Image_Processing.jpeg', caption="Poster Presentation")
    with col2:
        st.markdown("**Image Processing/Data Science Intern at UW Hospital (Medical Physics X-Ray Imaging Lab)**")
        st.write("""
            - **Objective**: Quantified the levels of endogenous Protoporphyrin-IX (PPIX) in skin pores to enhance the effectiveness of Photodynamic Therapy (PDT) for skin conditions and cancers.
            - **Data Processing**: Utilized **MATLAB** for data preprocessing and image analysis, applying techniques such as **Fourier Filtering** to isolate and enhance key features.
            - **Machine Learning**: Employed **k-means clustering** to segment regions of the skin based on PPIX concentration levels, allowing for the identification of high-risk areas.
            - **Statistical Analysis**: Performed **t-tests** to determine statistical significance in treatment efficacy before and after PDT intervention.
            - **Outcome**: The data-driven approach led to more accurate predictions of PPIX distribution, improving the precision of PDT treatment.
        """)
def projects():
     # Main Project: India Elections Data Analysis
    st.markdown("<h2 style='margin-bottom: 20px;'>Projects </h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display an image for the project
        st.image('India_Elections.png', caption="India Elections Data Analysis")
    with col2:
        st.markdown("**India Elections Data Analysis**")
        st.write("""
            - **GitHub**: [India Elections Data Analysis](https://github.com/asaigal2/India_Elections_Data_Analysis)
            - **Overview**: An in-depth data analysis project examining Indian election data, focusing on trends, geographical impact, and voting patterns.
            - **Tools Used**: Python (Pandas, Matplotlib), PostgreSQL for data management, and Streamlit for creating interactive visualizations.
            - **Key Features**:
                - Built interactive visualizations, like treemaps, histograms, stacked bar charts including a geo-spatial map of India with filters for exploring election metrics, using D3.js.
                - Analyzed historical data to predict future voting trends.
                - Containerised visualisations using streamlit library in Python and Django Framework.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section for Mini Projects
    st.markdown("<h3 style='margin-bottom: 20px;'>Mini Projects</h2>", unsafe_allow_html=True)

    # Mini Project 1: Wine Quality Classification
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('WineClassification.png', caption="Wine Quality Classification")
    with col2:
        st.markdown("**Project 1: Wine Quality Classification**")
        st.write("""
            - **GitHub**: [Wine Quality Classification](https://github.com/asaigal2/Mini-Projects/tree/main/Project1)
            - **Overview**: A machine learning classification project aimed at predicting the quality of white wine based on chemical properties.
            - **Dataset**: UCI Machine Learning Repository with approximately 5000 variations of white wine.
            - **Key Features**:
                - Cleaned data to handle missing values, normalized features, and prepared it for modeling.
                - **Models Used**: k-Nearest Neighbors (k-NN), Decision Tree, Random Forest, and Stochastic Gradient Descent (SGD) classifiers.
                - The dataset was split into training (80%) and testing (20%) sets.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Mini Project 2: NVDA Stock Prediction Model
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('NVDAstock.png', caption="NVDA Stock Prediction Model")
    with col2:
        st.markdown("**Project 2: NVDA Stock Prediction Model**")
        st.write("""
            - **GitHub**: [NVDA Stock Prediction](https://github.com/asaigal2/Mini-Projects/tree/main/Project2)
            - **Overview**: This project uses historical stock data to predict future prices of NVIDIA (NVDA) stocks.
            - **Dataset**: Historical stock data from Yahoo Finance.
            - **Key Features**:
                - Linear regression model trained and tested to predict stock prices.
                - The Jupyter Notebook contains data analysis, model training, and evaluation steps.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Mini Project 3: Exploratory Data Analysis
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('eda.png', caption="Exploratory Data Analysis/ Churn Analysis")
    with col2:
        st.markdown("**Exploratory Data Analysis (EDA) Project**")
        st.write("""
        -**GitHub**: [Churn Analysis](https://github.com/asaigal2/Practice-EDA)
        - **Tools Used**: Python (Pandas, Matplotlib, Seaborn, Scikit-learn).
        - **Overview**:
            1. Perform exploratory data analysis (EDA) to identify key factors influencing customer churn.
            2. Preprocess the data by handling categorical variables, missing values, and outliers.
            3. Build a Random Forest classifier to predict customer churn.
            4. Identify the top 20 customers most likely to churn based on model predictions.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Mini Project 4: Power BI Dashboard
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('PowerBI.png', caption="Power BI Dashboard")
    with col2:
        st.markdown("**Power BI Dashboard**")
        st.write("""
        - **Tools Used**: Power BI, Excel, DAX (Data Analysis Expressions).
        - **Overview**:
            - **KPIs**: Displayed high-level KPIs, including total sales ($12.64M), total profit ($1.47M), total quantity sold (178K units), and shipping cost ($1.35M).
            - **Geographical Breakdown**: Visualized sales distribution across different countries (e.g., USA, Germany, France), and further drilled down by regions and states (e.g., England, California).
            - **Sales by Category and Market**: Analyzed sales performance by product categories (Technology, Furniture, Office Supplies) and markets (APAC, EU, US).
            - **Shipping Analysis**: Explored shipping methods and their impact on sales, with Standard Class contributing the highest sales ($7.58M).
            - **Interactivity**: Enabled users to apply filters on product categories, regions, and shipping methods for dynamic data exploration.
            - **DAX Measures**: Used DAX to calculate metrics such as total sales, profit margins, and year-over-year growth, providing deeper insights into business performance.
    """)
def skills():
    st.title("Skills and Classes Taken")
    

    # Licenses & Certifications section
    st.subheader("Licenses & Certifications")
    st.write("""
    - **BCG - Data Science Job Simulation**  
    Issued by Forage (July 2024)  
    Credential ID: DjQM5itStfoaEyjpi  
    [View Credential](https://forage-uploads-prod.s3.amazonaws.com/completion-certificates/BCG%20/Tcz8gTtprzAS4xSoK_BCG_h9Y946hLET6T7PfhX_1721036137358_completion_certificate.pdf)
    )  

    - **Fundamentals of Analytics on AWS**  
      Issued by Amazon Web Services (AWS) (July 2024)  
      [View Credential](AWS_Skill_Builder_Certificate.pdf) 

    - **SQL**  
      Issued by HackerRank (July 2024)  
      [View Credential](https://www.hackerrank.com/certificates/iframe/07731f252abc)  
             
    - **Deloitte Salesforce Bootcamp** (Among 30 students accepted to gain hands-on Salesforce)

    - **Tableau**  
      Issued by Udemy
    """)

    # Classes section
    st.subheader("Classes Taken")
    st.write("""
    - **Calculus**
    - **Image Processing**
    - **Signal Processing**
    - **Probability & Statistics**
    - **Linear Algebra**
    - **Random Variables and Probability**
    - **Machine Learning**
    - **Matrix Methods in Machine Learning**
    - **Digital System Fundamentals**
    """)




# Create horizontal buttons using st.columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('About Me'):
        st.session_state.section = 'About Me'

with col2:
    if st.button('Technical Experience'):
        st.session_state.section = 'Technical Experience'

with col3:
    if st.button('Projects'):
        st.session_state.section = 'Projects'

with col4:
    if st.button('Skills/Classes Taken'):
        st.session_state.section = 'Skills/Classes Taken'

# Render the selected section based on the session state
if st.session_state.section == 'About Me':
    render_about_me()
elif st.session_state.section == 'Technical Experience':
    render_research_experience()
elif st.session_state.section == 'Projects':
    projects()
elif st.session_state.section == 'Skills/Classes Taken':
    skills()