import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool

# Load environment variables (for default values)
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="Python Code Tester Agent", layout="wide")
st.title("Python Code Tester Agent")

# API Key input in sidebar
st.sidebar.header("API Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
langchain_api_key = st.sidebar.text_input("LangChain API Key", type="password")

# Other configuration inputs
langchain_tracing_v2 = st.sidebar.checkbox("Enable LangChain Tracing V2")
langchain_endpoint = st.sidebar.text_input("LangChain Endpoint", value="https://api.smith.langchain.com")
langchain_project = st.sidebar.text_input("LangChain Project")

# Validate OpenAI API key
if not openai_api_key:
    st.error("Please enter your OpenAI API Key.")
    st.stop()

# Initialize OpenAI model
try:
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        openai_api_key=openai_api_key
    )

    # Define tools
    tools = [
        Tool(
            name="Python Interpreter",
            func=PythonREPLTool().run,
            description="Executes Python code snippets and returns the result. Useful for testing Python logic.",
        ),
    ]

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Set LangChain environment variables
    if langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = langchain_project

    # User input
    user_input = st.text_area("Enter your question or code to test:", "")

    if st.button("Submit"):
        if user_input.strip():
            st.info("Processing your query...")
            try:
                response = agent.run(user_input)
                formatted_response = response.replace("\\n", "\n")
                st.success("Agent Response:")
                st.markdown(f"### Result:\n{formatted_response}")
            except Exception as e:
                st.error(f"Error processing: {e}")
        else:
            st.warning("Please enter a query before submitting.")

except Exception as e:
    st.error(f"Error initializing the model: {e}")

st.markdown("""
### Features:
- **LangSmith Integration** for tracking queries and responses.
- **Python Code Execution** with a REPL tool.
- **OpenAI Models** with support for GPT-4, GPT-4o and GPT-4o-mini

### Security Note:
- Your API keys are used only for this session and are not stored.
- Ensure to keep your API keys private and do not share them.
""")

# Instructions for obtaining API keys
st.sidebar.markdown("""
### How to obtain API keys:
1. OpenAI API Key: [OpenAI Platform](https://platform.openai.com/signup)
2. LangChain API Key: [LangChain](https://www.langchain.com/)
""")
