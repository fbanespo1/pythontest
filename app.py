import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI, BedrockChat
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="Multi-Model Python Code Tester", layout="wide")
st.title("Multi-Model Python Code Tester")

# Model selection
model_option = st.sidebar.selectbox(
    "Choose Language Model",
    ("OpenAI GPT-4", "AWS Bedrock Claude", "AWS Bedrock Mistral", "AWS Bedrock Llama")
)

# API Key inputs
st.sidebar.header("API Configuration")
if model_option == "OpenAI GPT-4":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:  # AWS Bedrock models
    aws_access_key = st.sidebar.text_input("AWS Access Key", type="password")
    aws_secret_key = st.sidebar.text_input("AWS Secret Key", type="password")
    aws_region = st.sidebar.text_input("AWS Region", value="us-east-1")

# Debug mode
debug_mode = st.sidebar.checkbox("Enable Debug Mode")

# Initialize the selected model
try:
    if model_option == "OpenAI GPT-4":
        if not api_key:
            st.error("Please enter your OpenAI API Key.")
            st.stop()
        llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=api_key)
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
        elif model_option == "AWS Bedrock Llama":
            model_id = "meta.llama2-70b-chat-v1"
            model_kwargs = {"temperature": 0.1, "max_gen_len": 500}
        
        llm = BedrockChat(model_id=model_id, region_name=aws_region, model_kwargs=model_kwargs)

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
                    if model_option == "OpenAI GPT-4":
                        response = agent.run(user_input)
                    else:  # AWS Bedrock models
                        messages = [HumanMessage(content=f"Analyze and fix this Python code: {user_input}")]
                        try:
                            response = llm(messages)
                            if debug_mode:
                                st.text("Raw Response:")
                                st.code(str(response), language="text")
                            
                            if hasattr(response, 'content'):
                                response = response.content
                            elif isinstance(response, dict) and 'content' in response:
                                response = response['content']
                            else:
                                response = str(response)
                            
                            if not response.strip():
                                st.warning("The model returned an empty response. This might be due to API limitations or the complexity of the query.")
                        except Exception as e:
                            st.error(f"Error with Bedrock model: {str(e)}")
                            if debug_mode:
                                st.exception(e)
                            response = ""
                    
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

st.markdown("""
### Features:
- **Multiple Language Models**: Choose between OpenAI GPT-4, AWS Bedrock Claude, Mistral, and Llama.
- **Python Code Analysis**: Get insights and fixes for your Python code.
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
""")
