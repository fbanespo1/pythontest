import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool

# Configurazione di Streamlit
st.set_page_config(page_title="Python Code Tester Agent", layout="wide")
st.title("Python Code Tester Agent")

# Input per la chiave API
openai_api_key = st.sidebar.text_input("Inserisci la tua OpenAI API Key", type="password")

if openai_api_key:
    # Inizializzazione del modello OpenAI
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        openai_api_key=openai_api_key
    )

    # Definizione degli strumenti
    tools = [
        Tool(
            name="Python Interpreter",
            func=PythonREPLTool().run,
            description="Esegue frammenti di codice Python e restituisce il risultato.",
        ),
    ]

    # Inizializzazione dell'agente
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Interfaccia utente
    user_input = st.text_area("Inserisci la tua domanda o il codice da testare:", "")

    if st.button("Invia"):
        if user_input.strip():
            st.info("Elaborazione della tua richiesta...")
            try:
                response = agent.run(user_input)
                st.success("Risposta dell'Agente:")
                st.markdown(f"### Risultato:\n{response}")
            except Exception as e:
                st.error(f"Errore durante l'elaborazione: {e}")
        else:
            st.warning("Per favore, inserisci una query prima di inviare.")
else:
    st.warning("Per favore, inserisci la tua OpenAI API Key nella barra laterale per iniziare.")

st.markdown("""
### Funzionalità:
- **Esecuzione di Codice Python** con uno strumento REPL.
- **Modelli OpenAI** con supporto per GPT-4.
""")
