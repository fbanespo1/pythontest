import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI, BedrockChat
from langchain_community.llms import HuggingFaceHub
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.schema import HumanMessage
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="Multi-Model Python Code Tester", layout="wide")
st.title("Multi-Model Python Code Tester")

# Model selection
model_option = st.sidebar.selectbox(
    "Choose Language Model",
    ("OpenAI GPT-4", "AWS Bedrock (Claude)", "Mistral", "Llama")
)

# API Key inputs
st.sidebar.header("API Configuration")
if model_option == "OpenAI GPT-4":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
elif model_option == "AWS Bedrock (Claude)":
    aws_access_key = st.sidebar.text_input("AWS Access Key", type="password")
    aws_secret_key = st.sidebar.text_input("AWS Secret Key", type="password")
    aws_region = st.sidebar.text_input("AWS Region", value="us-east-1")
elif model_option in ["Mistral", "Llama"]:
    huggingface_api_key = st.sidebar.text_input("HuggingFace API Key", type="password")

# Initialize the selected model
try:
    if model_option == "OpenAI GPT-4":
        if not api_key:
            st.error("Please enter your OpenAI API Key.")
            st.stop()
        llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=api_key)
    elif model_option == "AWS Bedrock (Claude)":
        if not aws_access_key or not aws_secret_key:
            st.error("Please enter your AWS credentials.")
            st.stop()
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
        llm = BedrockChat(model_id="anthropic.claude-v2", region_name=aws_region)
    elif model_option == "Mistral":
        if not huggingface_api_key:
            st.error("Please enter your HuggingFace API Key.")
            st.stop()
        llm = HuggingFaceHub(repo_id="mistralai/Mamba-Codestral-7B-v0.1", huggingfacehub_api_token=huggingface_api_key)
    elif model_option == "Llama":
        if not huggingface_api_key:
            st.error("Please enter your HuggingFace API Key.")
            st.stop()
        llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.1-8B", huggingfacehub_api_token=huggingface_api_key)

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
        verbose=True,
        handle_parsing_errors=True
    )

    # User input
    user_input = st.text_area("Enter your Python code or question:", "")

    if st.button("Submit"):
        if user_input.strip():
            st.info(f"Processing your query using {model_option}...")
            try:
                with st.spinner("Analyzing..."):
                    if model_option == "AWS Bedrock (Claude)":
                        # Direct call to the model for Bedrock
                        response = llm([HumanMessage(content=f"Analyze and fix this Python code: {user_input}")])
                        st.success("Model Response:")
                        st.markdown(f"### Result:\n{response.content}")
                    else:
                        # Use the agent for other models
                        response = agent.run(user_input)
                        st.success("Agent Response:")
                        st.markdown(f"### Result:\n{response}")
            except Exception as e:
                st.error(f"Error processing: {str(e)}")
        else:
            st.warning("Please enter a query before submitting.")

except Exception as e:
    st.error(f"Error initializing the model: {str(e)}")

st.markdown("""
### Features:
- **Multiple Language Models**: Choose between OpenAI GPT-4, AWS Bedrock (Claude), Mistral, and Llama.
- **Python Code Execution**: Test and debug Python code snippets.
- **Flexible Querying**: Ask questions or input code for analysis.

### Security Note:
- Your API keys are used only for this session and are not stored.
- Ensure to keep your API keys private and do not share them.
""")

# Instructions for obtaining API keys
st.sidebar.markdown("""
### How to obtain API keys:
- OpenAI API Key: [OpenAI Platform](https://platform.openai.com/signup)
- AWS Credentials: [AWS Console](https://aws.amazon.com/)
- HuggingFace API Key: [HuggingFace](https://huggingface.co/settings/tokens)
""")
