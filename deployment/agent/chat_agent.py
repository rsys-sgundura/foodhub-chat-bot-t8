# 12.5
# ================================================================
#  FILE: chat_agent.py
# ---------------------------------------------------------------
#  FoodHub Conversational Assistant (Groq-exclusive version)
# ---------------------------------------------------------------
#  PURPOSE:
#   - Handles all user-facing chat interactions for FoodHub.
#   - Uses Groq-hosted LLaMA 4 model for short (<80 words), polite,
#     and context-aware responses.
#   - Detects intent (promo, refund, handoff, farewell, etc.)
#     and responds accordingly.
#   - Enforces data privacy and safety policies.
# ================================================================

import os
import re
import streamlit as st
from langchain_groq import ChatGroq


# ================================================================
#  SECTION 1: LLM Constructor (Groq Only)
# ---------------------------------------------------------------
#  Creates and configures a Groq-based LLaMA model instance.
#  This model is used for all polite and context-aware responses.
# ================================================================
def make_refiner(temperature: float = 0.7):
    """
    Create a ChatGroq model instance for refined conversational responses.
    """

    # Step 1: Retrieve API key from environment
    # (Expected to be set externally for secure authentication)
    #groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable for LLM access.")

    # Step 2: Initialize Groq LLM with given temperature setting
    # - Higher temperature = more creative / varied responses
    # - Lower temperature = deterministic, factual responses
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=groq_api_key,
        temperature=temperature,
    )


# ================================================================
#  SECTION 2: Intent Patterns
# ---------------------------------------------------------------
#  Defines keyword-based regular expressions to detect
#  user intent categories like:
#   - Promo, Refund, Handoff, Farewell, and Enumeration
#  These help route the message to the appropriate logic path.
# ================================================================
_PATTERNS = {
    "promo": [
        r"\bpromo(?:tion|tions)?\b", r"\boffer(?:s)?\b", r"\bdiscount\b",
        r"\bdeal(?:s)?\b", r"\bcoupon(?:s)?\b", r"\bsale\b"
    ],
    "farewell": [
        r"\bthank(s| you| u)\b", r"\bbye\b", r"\bsee you\b",
        r"\btake care\b", r"\bend chat\b"
    ],
    "refund": [
        r"\brefund\b", r"\breturn\b", r"\bcancel\b", r"\bcancellation\b"
    ],
    "handoff": [
        r"\bconnect (to|with) (human|agent)\b",
        r"\bchat with (a )?(person|agent|executive)\b",
        r"\btalk to (human|representative)\b",
        r"\bcustomer support\b"
    ],
    "enumeration": [
        r"\ball\b", r"\bevery\b", r"\bcomplete\b",
        r"\bselect \*\b", r"\bexport\b", r"\bdatabase\b"
    ],
}


def _match(patterns, text: str) -> bool:
    """
    Utility function to check if the user's message
    matches any given list of regular expression patterns.
    """
    text = (text or "").lower()
    return any(re.search(p, text) for p in patterns)


# ================================================================
#  SECTION 3: Prompt Builders
# ---------------------------------------------------------------
#  These functions construct custom prompts for the LLM
#  based on user intent type (e.g., Promo or General inquiry).
#  Each ensures responses are polite, concise, and policy-safe.
# ================================================================

def promo_prompt(user_query: str, cust_id: str) -> str:
    """Constructs prompt for promotions/offers."""
    name = cust_id or "there"
    return f"""
You are FoodHub’s courteous assistant.
Respond to {name} asking about promotions.
Keep replies under 80 words — friendly, clear, and professional.
Mention one or two current offers such as "10% off meals above ₹500" or "Free weekend delivery".
End with a gentle follow-up like “Would you like me to apply this?”.
User query: {user_query}
""".strip()


def general_prompt(user_query: str, cust_id: str) -> str:
    """Constructs prompt for general, non-promo user messages."""
    name = cust_id or "there"
    return f"""
You are FoodHub’s helpful chat assistant.
Rules:
- Write a short, natural message (<80 words), warm but concise.
- Address {name} personally.
- Do not expose IDs, logs, or database details.
- If user asks for ‘all customers’ or entire data, refuse politely citing privacy policy.
User query: {user_query}
""".strip()


# ================================================================
#  SECTION 4: Main Chat Handler
# ---------------------------------------------------------------
#  Core logic that routes user messages to the correct intent flow:
#   1. Farewell
#   2. Refund
#   3. Human Handoff
#   4. Privacy Enforcement
#   5. LLM-based Response Generation
#  Ensures all outputs are polite, safe, and relevant.
# ================================================================
def answer_tool_func(answertool_input: str) -> str:
    """
    Receives the output from OrderQueryTool as stringified dict,
    Handles incoming chat queries, detect intent, and generate polite responses.
    Routes queries through LLM or fixed policy paths.
    """
    try:
        data = ast.literal_eval(answertool_input)
        cust_id = data.get("cust_id", "Unknown")
        orig_query = data.get("orig_query", "")
        db_orders = data.get("db_orders", "No order details found.")
    except Exception:
        return "⚠️ Error: Could not parse order data properly."


    query = (db_orders or "").strip().lower()

    # Initialize the model with default temperature (moderately creative)
    llm = make_refiner()

    # ---------------------------------------------------------------
    # STEP 1: Handle Farewell Intent
    # ---------------------------------------------------------------
    if _match(_PATTERNS["farewell"], query):
        # Prevent repeated goodbye messages within same session
        #if not st.session_state.get("said_goodbye", False):
            #st.session_state["said_goodbye"] = True
            name = cust_id or "there"
            return f"Thanks, {name}! It was lovely assisting you today. Have a great day ahead!"
        #return ""

    # ---------------------------------------------------------------
    # STEP 2: Handle Refund or Cancellation Intent
    # ---------------------------------------------------------------
    if _match(_PATTERNS["refund"], query):
        return (
            "I understand, and I’ve shared your refund or cancellation request with our human team. "
            "They’ll verify and get back to you shortly via email. Thank you for your patience."
        )

    # ---------------------------------------------------------------
    # STEP 3: Handle Human Handoff Requests
    # ---------------------------------------------------------------
    if _match(_PATTERNS["handoff"], query):
        return (
            "Sure! I’ll connect you with a human support specialist right away. "
            "Please hold while I pass your request along."
        )

    # ---------------------------------------------------------------
    # STEP 4: Enforce Privacy Policy
    # ---------------------------------------------------------------
    if _match(_PATTERNS["enumeration"], query):
        # Prevent requests for all data or database export
        return (
            "Sorry, I can’t share complete data or all customer details due to our privacy policy. "
            "Please ask about your specific account or order instead."
        )

    # ---------------------------------------------------------------
    # STEP 5: Select Prompt Type (Promo or General)
    # ---------------------------------------------------------------
    if _match(_PATTERNS["promo"], query):
        prompt = promo_prompt(db_orders, cust_id)
    else:
        prompt = general_prompt(db_orders, cust_id)

    # ---------------------------------------------------------------
    # STEP 6: Generate Response from LLM
    # ---------------------------------------------------------------
    try:
        # Invoke the model with the constructed prompt
        raw_out = llm.invoke(prompt)
        response = getattr(raw_out, "content", str(raw_out)).strip()

        # If response is blank or uncertain, trigger fallback
        if not response or any(k in response.lower() for k in ["not sure", "don’t know", "cannot answer"]):
            return (
                "I’m not entirely confident about that. I’ve asked our human support team to review this "
                "and follow up with you soon."
            )

        # -----------------------------------------------------------
        # Store conversation context in session for smoother replies
        # -----------------------------------------------------------
        if "promotion" in response.lower():
            #st.session_state["last_context"] = "promo"
            last_ctx = "promo"
        elif "order" in response.lower():
            #st.session_state["last_context"] = "order"
            last_ctx = "order"
        else:
            #st.session_state["last_context"] = ""
            last_ctx = ""
        # -----------------------------------------------------------
        # Handle short acknowledgements (ok / thanks)
        # -----------------------------------------------------------
        if query in {"ok", "okay", "thanks", "thank you"}:
            name = cust_id or "there"
            #last_ctx = st.session_state.get("last_context", "")
            if last_ctx == "promo":
                return f"You’re welcome, {name}! Want me to apply that offer to your next order?"
            if last_ctx == "order":
                return f"Glad to help, {name}! I’ll keep you updated on your order progress."
            return f"You’re most welcome, {name}!"

        # -----------------------------------------------------------
        # Return the final refined response
        # -----------------------------------------------------------
        return response

    # ---------------------------------------------------------------
    # STEP 7: Exception Handling
    # ---------------------------------------------------------------
    except Exception as e:
        # Fallback in case of LLM or API issues
        return (
            "Apologies, something went wrong while processing your request. "
            "Our human team has been notified and will assist you shortly."
        )


# ================================================================
#  SECTION 5: LangChain Tool Wrapper
# ---------------------------------------------------------------
#  Wraps the chat handler as a LangChain Tool so that it can be
#  called within multi-agent workflows or pipelines.
# ================================================================
#AnswerTool = Tool(
#    name="answer_tool",
#    func=answer_tool_func,
#    description="Format raw DB results into a brief, polite user-facing message. Enforces business rules (cancelled/completed messaging, escalation)."
#)
