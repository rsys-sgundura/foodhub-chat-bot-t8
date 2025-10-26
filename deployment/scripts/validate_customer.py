
import re
from agent.sql_agent import db_agent_executor

# simple validators
def extract_cust_id(text: str):
    """Return cust_id in format C#### or None"""
    m = re.search(r"\b(C\d{4})\b", text, flags=re.I)
    return m.group(1).upper() if m else None

# Validate customer ID using the SQL Agent
def is_valid_customer(customer_id: str) -> bool:
    if not extract_cust_id(customer_id):
        return False
    try:
        query = f"Check if customer ID {customer_id} exists in the database"
        response = db_agent_executor.run(query)
        return "does not exist" not in response.lower()
    except Exception as e:
        print(f"Agent error: {e}")
        return False
