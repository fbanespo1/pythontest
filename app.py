import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI, BedrockChat
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.schema import HumanMessage
from langchain.callbacks import StreamlitCallbackHandler
from langfuse import Langfuse
from langfuse.decorators import observe
from langfuse.openai import openai  # OpenAI integration

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="Multi-Model Python Code Tester", layout="wide")
st.title("Multi-Model Python Code Tester")

# Model selection
model_option = st.sidebar.selectbox(
    "Choose Language Model",
    ("OpenAI GPT-4", "AWS Bedrock Claude", "AWS Bedrock Mistral")
)

# API Key inputs
st.sidebar.header("API Configuration")
if model_option == "OpenAI GPT-4":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:  # AWS Bedrock models
    aws_access_key = st.sidebar.text_input("AWS Access Key", type="password")
    aws_secret_key = st.sidebar.text_input("AWS Secret Key", type="password")
    aws_region = st.sidebar.text_input("AWS Region", value="us-east-1")

# LangSmith configuration
langchain_api_key = st.sidebar.text_input("LangChain API Key", type="password")
langchain_project = st.sidebar.text_input("LangChain Project", value="default")
langchain_tracing_v2 = st.sidebar.checkbox("Enable LangChain Tracing V2")

# Langfuse configuration
st.sidebar.header("Langfuse Configuration")
langfuse_public_key = st.sidebar.text_input("Langfuse Public Key", type="password")
langfuse_secret_key = st.sidebar.text_input("Langfuse Secret Key", type="password")
langfuse_host = st.sidebar.text_input("Langfuse Host", value="https://cloud.langfuse.com")

# Debug mode
debug_mode = st.sidebar.checkbox("Enable Debug Mode")

# Initialize Langfuse
if langfuse_public_key and langfuse_secret_key:
    langfuse = Langfuse(
        public_key=langfuse_public_key,
        secret_key=langfuse_secret_key,
        host=langfuse_host
    )
else:
    langfuse = None

# Initialize the selected model
try:
    if model_option == "OpenAI GPT-4":
        if not api_key:
            st.error("Please enter your OpenAI API Key.")
            st.stop()
        openai.api_key = api_key  # Set OpenAI API key

        @observe()
        def get_openai_response(user_input):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                temperature=0
            )
            return response.choices[0].message.content

    else:  # AWS Bedrock models
        if not aws_access_key or not aws_secret_key:
            st.error("Please enter your AWS credentials.")
            st.stop()
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key

        if model_option == "AWS Bedrock Claude":
            model_id = "anthropic.claude-v2"
            model_kwargs = {"temperature": 0.1, "max_tokens_to_sample": 500}
        elif model_option == "AWS Bedrock Mistral":
            model_id = "mistral.mistral-7b-instruct-v0:2"
            model_kwargs = {"temperature": 0.1, "max_tokens": 500}

        llm = BedrockChat(model_id=model_id, region_name=aws_region, model_kwargs=model_kwargs)

        @observe()
        def get_bedrock_response(user_input):
            messages = [HumanMessage(content=f"Analyze and fix this Python code: {user_input}")]
            response = llm(messages)
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            else:
                return str(response)

    # Define tools
    tools = [
        Tool(
            name="Python Interpreter",
            func=PythonREPLTool().run,
            description="Executes Python code snippets and returns the result. Useful for testing Python logic.",
        ),
    ]

    # Initialize agent (only for OpenAI)
    if model_option == "OpenAI GPT-4":
        agent_llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=api_key)
        agent = initialize_agent(
            tools=tools,
            llm=agent_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    # Set LangChain environment variables
    if langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = langchain_project

    # User input
    user_input = st.text_area("Enter your Python code or question:", "")

    if st.button("Submit"):
        if user_input.strip():
            st.info(f"Processing your query using {model_option}...")
            try:
                with st.spinner("Analyzing..."):
                    if model_option == "OpenAI GPT-4":
                        if 'agent' in locals():
                            # Use the agent if initialized
                            response = agent.run(user_input)
                        else:
                            response = get_openai_response(user_input)
                    else:  # AWS Bedrock models
                        response = get_bedrock_response(user_input)

                    if response.strip():
                        st.success("Model Response:")
                        st.markdown(f"### Result:\n{response}")
                    else:
                        st.warning("The model did not provide a response. This might be due to API limitations or the complexity of the query.")
            except Exception as e:
                st.error(f"Error processing: {str(e)}")
                if debug_mode:
                    st.exception(e)
        else:
            st.warning("Please enter a query before submitting.")

except Exception as e:
    st.error(f"Error initializing the model: {str(e)}")
    if debug_mode:
        st.exception(e)
