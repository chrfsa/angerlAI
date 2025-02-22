import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import altair as alt
from langchain_community.vectorstores import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import dotenv
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings


# Charger les variables d'environnement
dotenv.load_dotenv()

# Fonction pour la conversation avec l'assistant
def chat(question):
    REVIEWS_CHROMA_PATH = "chroma_data/"

    # Modèle de message système
    system_template_str = """
    You are an advanced language model assistant. Utilize the provided context to generate precise and relevant responses, incorporating factual data and statistics where appropriate.
    repondre avec des poids precis,
    en utilisant le nombre de visiteur, les croissance financieres et les invistisseur et tous.
    Context:
    {context}
    """


    review_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["context"], template=system_template_str)
    )

    # Modèle de message humain
    human_template_str = """
    The user has a question. Provide a detailed and helpful response based on the provided context, focusing primarily on numerical data and statistics.

    Question: {question}
    Please respond to this question in the same language.
    Provide a detailed answer, justifying your responses with numerical evidence and examples.
    Avoid providing information that is not explicitly mentioned in the CONTEXT INFORMATION.
    Do not use phrases such as "according to the context" or "mentioned in the context" or "the information provided in the context".
    """


    review_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["question"], template=human_template_str)
    )

    # Liste des messages à échanger
    messages = [review_system_prompt, review_human_prompt]

    review_prompt_template = ChatPromptTemplate(input_variables=["context", "question"], messages=messages)

    # Modèle de l'assistant de chat
    chat_model = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    # Analyseur de sortie
    output_parser = StrOutputParser()

    # Base de données de vecteurs de critiques
    reviews_vector_db = Chroma(persist_directory=REVIEWS_CHROMA_PATH, embedding_function=CohereEmbeddings(model="embed-multilingual-v3.0"))

    reviews_retriever = reviews_vector_db.as_retriever(k=10)

    # Chaine de traitement
    review_chain = (
        {"context": reviews_retriever, "question": RunnablePassthrough()}
        | review_prompt_template
        | chat_model
        | output_parser
    )
    return review_chain.invoke(question)


# Afficher le logo en haut à gauche
st.sidebar.image("./logo.png", use_column_width=True)

# Initialiser l'état de redirection si ce n'est pas déjà fait
if "redirect" not in st.session_state:
    st.session_state.redirect = False

# Initialiser l'état de la réponse si ce n'est pas déjà fait
if "answer" not in st.session_state:
    st.session_state.answer = None

# Définir la barre de navigation
with st.sidebar:
    selected = option_menu(
            "Main Menu", 
            ["Chatbot", "Dashboard"], 
            icons=['robot', 'house'], 
            menu_icon="cast", 
            default_index=0,  # Définir 0 pour sélectionner Chatbot par défaut
        )

# Définir le composant Chatbot
def chatbot_component():
    st.title("AngelAI")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Afficher ou effacer les messages de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Fonction pour générer la réponse de LLaMA2

    # Prompt fourni par l'utilisateur
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Générer une nouvelle réponse si le dernier message n'est pas de l'assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                response = chat(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)


# Définir le composant Dashboard
import altair as alt

def dashboard_component():
    st.title("Courbe de Croissance des Startups")
    
    # Exemple de données de croissance des startups
    data = pd.DataFrame({
        'Année': ['2023', '2024', '2023', '2024', '2023'],
        'Startups': ['openAI', 'mistral ia', 'aCriteo', 'Snips', 'DeepMind'],
        'Croissance': [1.8, 2.0, 1.5, 2.1, 0.8]
    })
    
    # Afficher les données sous forme de tableau
    st.table(data)

    # Graphique Altair pour la courbe de croissance (Ligne)
    line_chart = alt.Chart(data).mark_line(point=True).encode(
        x='Année',
        y='Croissance',
        color='Startups',
        tooltip=['Startups', 'Année', 'Croissance']
    ).interactive()
    st.altair_chart(line_chart, use_container_width=True)

    # Graphique Altair pour les barres
    bar_chart = alt.Chart(data).mark_bar().encode(
        x='Année',
        y='Croissance',
        color='Startups',
        tooltip=['Startups', 'Année', 'Croissance']
    ).interactive()
    st.altair_chart(bar_chart, use_container_width=True)

    # Graphique Altair pour les secteurs (camemberts)
    pie_chart = alt.Chart(data).mark_arc().encode(
        theta=alt.Theta(field="Croissance", type="quantitative"),
        color=alt.Color(field="Startups", type="nominal"),
        tooltip=['Startups', 'Croissance']
    ).interactive()
    st.altair_chart(pie_chart, use_container_width=True)

    # Graphique Altair pour l'histogramme
    hist_chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Croissance:Q', bin=True),
        y='count()',
        color='Startups',
        tooltip=['Startups', 'Année', 'Croissance']
    ).interactive()
    st.altair_chart(hist_chart, use_container_width=True)

    # Graphique Altair pour la boîte à moustaches
    boxplot_chart = alt.Chart(data).mark_boxplot().encode(
        x='Startups',
        y='Croissance',
        color='Startups',
        tooltip=['Startups', 'Année', 'Croissance']
    ).interactive()
    st.altair_chart(boxplot_chart, use_container_width=True)

# Afficher le contenu en fonction de la sélection
if selected == "Chatbot":
    chatbot_component()
elif selected == "Dashboard":
    dashboard_component()
