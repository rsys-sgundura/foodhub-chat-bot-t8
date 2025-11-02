
import streamlit as st
import base64
from validate_customer import is_valid_customer
from agent.chat_agent import answer_tool_func
from agent.sql_agent import order_query_tool_func
import sys

# ================================================================
#  12.7: Chat Agent Initialization
# ---------------------------------------------------------------
#  PURPOSE:
#   - Combine all tools (OrderQueryTool + AnswerTool) into a unified
#     conversational agent for FoodHub.
#   - The agent can handle both database queries and chat responses.
#   - Uses a deterministic LLM model to maintain concise, accurate replies.
# ================================================================

# -------------------------------------------------------------------
#  Step 1: LangChain Tool Wrapper for Query Tool function
# -------------------------------------------------------------------
#  Wraps the SQL query executor as a callable Tool.
#  Enables integration with agent workflows that need database access.
# ===================================================================
from langchain.tools import Tool

OrderQueryTool = Tool(
    name="order_query_tool",
    func=order_query_tool_func,
    description="Use this tool to fetch order-related (read-only) info for a customer's order. Requires customer id from session. Blocks confidential fields. Returns structured output as a stringified dictionary"
)

# -----------------------------------------------------------------
#  Step 2: LangChain Tool Wrapper for Answer Tool function
# -----------------------------------------------------------------
#  Wraps the chat handler as a LangChain Tool so that it can be
#  called within multi-agent workflows or pipelines.
# ================================================================
AnswerTool = Tool(
    name="answer_tool",
    func=answer_tool_func,
    description="Format raw DB results into a brief, polite user-facing message. Enforces business rules (cancelled/completed messaging, escalation)."
)

# ------------------------------------------------
# Step 3: Register active LangChain tools
# ------------------------------------------------
#  These tools define the functional abilities of the agent.
#  - OrderQueryTool : Handles order-related SQL lookups
#  - AnswerTool     : Generates friendly natural-language replies
tools = [OrderQueryTool, AnswerTool]

# ------------------------------------------------
# Step 4: Initialize the Chat Agent
# ------------------------------------------------
#  Creates a ZERO_SHOT_REACT_DESCRIPTION type agent:
#   - Zero-shot reasoning: Understands and reacts without prior examples
#   - React-style: Uses reasoning traces internally to plan actions
#   - Description-based: Uses each tool’s description for action selection
chat_agent = initialize_agent(
    tools=tools,                                # Registered functional tools
    llm=llm_model,                              # Underlying LLM (Groq-based)
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent reasoning type
    verbose=False                               # Suppress detailed logs for cleaner output
)

# ================================================================
#  Agent Controller
# ---------------------------------------------------------------
#  This section defines the controller functions that manage
#  conversation flow between the user and the toolchain.
#  The controller ensures the correct sequential execution of:
#     1. OrderQueryTool → retrieves database results
#     2. AnswerTool → formats and finalizes user-facing answers
# ================================================================


# ------------------------------------------------
# Step 5: Define agent_response()
# ------------------------------------------------
#  - Acts as the main orchestrator for tool execution.
#  - Builds an internal instruction prompt that enforces a strict
#    sequence of actions for the agent to follow.
#  - Ensures the pipeline runs deterministically:
#       OrderQueryTool → AnswerTool
# ------------------------------------------------
def agent_response(cust_id: str, user_query: str) -> str:
    """
    Coordinates the sequential execution of tools: OrderQueryTool → AnswerTool.
    """
    
    # === Construct internal execution prompt ===
    # This prompt explicitly tells the LLM how to sequence tool usage.
    agent_prompt = f"""
    You are FoodHub’s Order Assistant.
    Follow this exact execution sequence:
    1️⃣ Invoke 'OrderQueryTool' using:
        input_string = str({{"cust_id": "{cust_id}", "user_query": "{user_query}"}})
      → Output: a stringified dictionary containing 'cust_id', 'orig_query', and 'db_orders'.
    2️⃣ Next, call 'AnswerTool' with:
        input_string = the output received from OrderQueryTool.
    3️⃣ Finally, return **only** the precise response produced by AnswerTool — without rephrasing, summarizing, or altering it.
    """

    # === Execute the agent prompt ===
    # The chat agent interprets the instructions and handles tool chaining internally.
    final_answer = chat_agent.run(agent_prompt)
    
    # === Return final tool-generated response ===
    # Output corresponds exactly to the AnswerTool’s result.
    return final_answer


# ------------------------------------------------
# Step 6: Define chatbot_response()
# ------------------------------------------------
#  - Public-facing function used by the chat interface.
#  - Accepts validated `cust_id` and natural-language `user_query`.
#  - Delegates execution to agent_response() for LLM-driven handling.
#  - Returns final formatted reply for user display.
# ------------------------------------------------
def chatbot_response(cust_id: str, user_query: str) -> str:

    # === Forward user request to the Agent Controller ===
    final_llm_response = agent_response(cust_id, user_query)

    # === Return LLM-processed chatbot reply ===
    return final_llm_response

# ================================================================
#  Streamlit UI CONFIGURATION
# ---------------------------------------------------------------
#  This section PREPARES THE UI APP
# ================================================================

# --- App Configuration ---
st.set_page_config(page_title="FoodHub Chatbot", page_icon="🍽️", layout="wide")

# --- Load Local JPG Background Image ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

image_base64 = get_base64_image("foodhub_background_jpg.jpg")  # Make sure this file exists

# --- Session State Initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.authenticated:
    # --- Inject CSS for Background ---
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{image_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
       """,
       unsafe_allow_html=True
    )
else:
    # Clear background after login
    st.markdown(f"""
        <style>
        /* Remove background from stApp so blurred layer shows through */
        .stApp {{
            background-image: none !important;
            background-color: transparent !important;
        }}
        .blurred-bg {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: url("data:image/jpeg;base64,{image_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            z-index: -1;
            filter: blur(6px);
        }}
        </style>
        <div class="blurred-bg"></div>
        """,
        unsafe_allow_html=True
    )


if not st.session_state.authenticated:
    # Login form
    col1, col2 = st.columns([2, 2]) # Adjust ratios for desired spacing

    with col2:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("foodhub_logo.png", width=500)
        with col2:
            st.markdown("<h1 style='color: #ff4b4b; padding-top: 10px;'>Welcome to FoodHub Chatbot</h1>", unsafe_allow_html=True)
        # Instructional message in black
        st.markdown("<p style='color: black; font-size: 16px;'>Please enter customer ID and password to continue</p>", unsafe_allow_html=True)
        with st.form("login_form"):

            # Labels in black using label_visibility workaround
            #st.markdown("<label style='color: black;'>Customer ID</label>", unsafe_allow_html=True)
            customer_id = st.text_input("Customer ID", placeholder="eg: C1018")

            #st.markdown("<label style='color: black;'>Password</label>", unsafe_allow_html=True)
            password = st.text_input("Password", type="password")

            submitted = st.form_submit_button("Login")

            if submitted:
                # Add your login logic here
                if is_valid_customer(customer_id) and password == "foodhub123":
                    st.session_state.authenticated = True
                    st.session_state.customer_id = customer_id
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

# --- Chatbot Interface ---
if st.session_state.authenticated:
    customer_id = st.session_state.get("customer_id")
    # Ensure chat history
    if not st.session_state.chat_history:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": f"Hi! How can I help you today?"}
        ]
    spacer_left, chat_col, spacer_right = st.columns([2, 4, 1])

    with chat_col:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.image("foodhub_logo.png", width=100)
        with col2:
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; justify-content: flex-start; height: 100%;'>
                    <h1 style='color: #ff4b4b; margin: 0;'>Hey {customer_id}, Welcome!</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown("<div style='text-align: right; padding-top: 10px;'>", unsafe_allow_html=True)
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.customer_id = None
                st.session_state.chat_history = []
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Inject custom CSS for chat bubbles
        st.markdown("""
        <style>
        .chat-bubble-user {
            background-color: #dfe6e9;  /* soft gray-blue, neutral on light/dark */
            color: #000000;             /* black text */
            padding: 10px 14px;
            border-radius: 12px;
            margin-bottom: 6px;
            display: inline-block;
            max-width: 80%;
            text-align: right;
        }

        .chat-bubble-bot {
            background-color: #f1f0f0;  /* light gray, works on both themes */
            color: #000000;             /* black text */
            padding: 10px 14px;
            border-radius: 12px;
            margin-bottom: 6px;
            display: inline-block;
            max-width: 80%;
            text-align: left;
        }
        </style>
        """, unsafe_allow_html=True)

        # Chat rendering
        for m in st.session_state.chat_history:
            left_col, right_col = st.columns([1, 1])

            if m["role"] == "user":
                with right_col:
                    st.markdown(
                        f"""
                        <div style='display: flex; justify-content: flex-end; align-items: center; margin-bottom: 8px;'>
                            <div class='chat-bubble-user'>{m['content']}</div>
                            <div style='font-size: 20px; margin-left: 8px;'>🙋</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:
                with left_col:
                    st.markdown(
                        f"""
                        <div style='display: flex; justify-content: flex-start; align-items: center; margin-bottom: 8px;'>
                            <div style='font-size: 20px; margin-right: 8px;'>🤖</div>
                            <div class='chat-bubble-bot'>{m['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # 2) input → append → bot → append → rerun
        user_input = st.chat_input("Ask about your order or menu...")
        if user_input:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            with st.spinner("Let me check that for you..."):
              # Step 1: Call the Chat Response tool
            final_response = chatbot_response(customer_id, user_input)

            print('Output of chatbot_response tool:',final_response, flush=True)
            sys.stdout.flush()

            st.session_state.chat_history.append({"role":"assistant","content":final_response})
            st.rerun()
