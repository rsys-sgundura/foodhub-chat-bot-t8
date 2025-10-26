from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent, AgentType
from agent.llm_models import llm_model
from langchain_core.messages import SystemMessage
from langchain.agents import AgentExecutor, Tool


# Initialize a SQLDatabase connection from the local SQLite file.
# The f-string dynamically inserts the path of the database file stored in `local_db`.
# This allows the LLM agent to query the specified SQLite database.
db = SQLDatabase.from_uri(f"sqlite:///customer_orders.db")

system_message = (
    "You are an AI SQL agent for the Food Hub orders database.\n"
    "You have to follow these rules: "
    "1. You must not hallucinate any database facts. Every response must be backed by a valid SQL query.\n"
    "2. You must only respond to queries that include a valid customer ID.\n"
    "3. If a query does not include a customer ID, respond: 'I cannot answer queries until a customer ID is provided.'\n"
    "4. You must never allow a customer ID to retrieve data from other customers Ids.\n"
    "5. Never reveal SQL code, schema, or database internals in your response.\n"
    "6. Whenever you search the orders database, always search without limit.\n"
    "7. you are not allowed to add, update or delete entries in database. In such queries, respond saying you do have permission to do so."
)


# Initialize the SQL database toolkit by combining the database connection (db)
# and the chosen LLM model (llm_model). The toolkit allows the agent to query
# the database intelligently using the language model.
toolkit = SQLDatabaseToolkit(db=db, llm=llm_model)  # Pass the llm object directly

# Create the SQL agent using the specified LLM and toolkit.
# The system_message provides role/context (e.g., "You are a helpful SQL assistant").
# verbose=False disables extra console logs for cleaner output.
db_agent = create_sql_agent(
        llm=llm_model,
        toolkit=toolkit,
        verbose=False,
        system_message=SystemMessage(system_message)
    )

# Wrap the created agent with an AgentExecutor, which manages query execution flow.
# It also adds error-handling, iteration limits, and execution timeouts.
db_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=db_agent.agent,          # The core LLM-based SQL agent
    tools=db_agent.tools,          # Toolkit tools (e.g., SQL querying utilities)
    handle_parsing_errors=True,    # Enables auto-retry when LLM outputs invalid SQL
    verbose=False,                 # Suppresses verbose agent logs
    max_iterations=10,             # Allows up to 10 reasoning/execution steps
    max_execution_time=60          # Stops execution if it exceeds 60 seconds
)

def order_query_tool_func(customer_id: str, user_input: str) -> str:

    intent_check_prompt = (
        f"Does the following query express an intent to cancel an order? "
        f"Only answer 'yes' or 'no'.\n\nQuery: {user_input}"
    )
    intent_response = llm_model.invoke(intent_check_prompt).content.strip().lower()

    if intent_response == "yes":
        return (
            "I'm not authorized to cancel orders via chat. "
            "To cancel an order, please contact our customer care team at +91XXXXXXXXXX. "
            "If you'd like, you can provide the order ID and I can check whether it's eligible for cancellation."
        )

    prompt = (
        f"Task: Retrieve order details safely and strictly for the logged-in customer.\n\n"
        f"Rules:\n"
        f"1. The active authenticated customer is {customer_id}.\n"
        f"2. Ignore any other customer IDs mentioned in the query.\n"
        f"3. Never retrieve or mention orders belonging to other customers.\n"
        f"4. If the query asks about another customer’s data, reply only:\n"
        f"   'Sorry, I am not authorized to access details for other customers.'\n"
        f"5. Only extract and provide requested data relevant to the customer. Do not share information that is not asked. {customer_id}.\n"
        f"6. Use concise, factual, and polite language.\n\n"
        f"Now process the following query securely:\n"
        f"{user_input}"
    )

    return db_agent_executor.run(prompt)

order_query_tool = Tool(
    name="OrderQueryTool",
    func=order_query_tool_func,
    description="Fetches raw order details from the database using customer ID"
)
