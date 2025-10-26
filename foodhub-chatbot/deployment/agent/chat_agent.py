import re
from agent.llm_models import llm_high_model
from langchain.agents import initialize_agent
from langchain.agents import Tool

# Answer Tool: Refines raw response into customer-friendly message
def answer_tool_func(raw_response: str) -> str:
    exit_keywords = ["exit", "stop", "quit", "goodbye", "bye", "end chat"]
    if any(keyword in raw_response.lower() for keyword in exit_keywords):
        farewell_prompt = (
            "The user is ending the chat session. Please respond with a warm, polite farewell message "
            "that thanks them for using the service and invites them to return if they need help again."
        )
        return llm_high_model.invoke(farewell_prompt).content

    # Normal response prompt
    prompt = (
        "You are a polite FoodHub customer support assistant.\n"
        "Rewrite the following raw order information into one concise, natural sentence.\n"
        "Respond directly with the final message — no labels, no explanations.\n"
        "Use friendly, professional tone suitable for a customer chat.\n"
        "If data indicates an issue or delay, express empathy briefly.\n"
        "If status is normal, keep the tone positive and reassuring.\n\n"
        "If status is normal, Please Share only requested information and with hold unasked information even if they are extracxted from the table."
        "Raw data:\n"
        f"{raw_response}"
    )
    return llm_high_model.invoke(prompt).content
