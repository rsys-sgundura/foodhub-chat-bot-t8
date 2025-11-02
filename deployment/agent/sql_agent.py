# 12.4
# ================================================================
#  FILE: sql_agent.py
#  MODULE: FoodHub Secure SQL Query Handler (Groq-exclusive)
# ---------------------------------------------------------------
#  PURPOSE:
#  Safely processes natural language queries into secure, read-only
#  SQL statements using Groq-powered deterministic LLM reasoning.
#
#  KEY FEATURES:
#  ✅ SELECT-only enforcement (no data modification)
#  ✅ Restricted to specific cust_id
#  ✅ Anti-enumeration and anti-destructive query filters
#  ✅ Dynamic schema inspection and caching
#  ✅ Deterministic (low-temperature) LLM for reproducibility
# ================================================================

import os
import re
import sqlite3
import textwrap
import traceback
import pandas as pd
import ast
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain_groq import ChatGroq


# ============================================================
# === 12.1 Define Helper Utilities and Safety Checks ===
# ============================================================
# These functions and constants help sanitize inputs, detect intent,
# and enforce data privacy before executing SQL or agent queries.
# ------------------------------------------------------------

import re
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import pandas as pd

# ------------------------------------------------------------
# STEP 1: Define confidential (sensitive) database columns
# ------------------------------------------------------------
# These columns must never be returned to the user.
# Use a try/except to avoid redefining if declared earlier.
#try:
#    confidential_columns  # Check if variable already exists
#except NameError:
confAidential_columns = ["prepared_time", "delivery_time"]

# ------------------------------------------------------------
# STEP 2: Utility — Extract Customer ID from text
# ------------------------------------------------------------
def extract_cust_id(text: str):
    """
    Identify and return a customer ID in the format 'C####' (e.g., C1011).
    Returns None if not found.
    """
    m = re.search(r"\b(C\d{4})\b", text, flags=re.I)
    return m.group(1).upper() if m else None

# ------------------------------------------------------------
# STEP 3: Utility — Extract Order ID from text
# ------------------------------------------------------------
def extract_order_id(text: str):
    """
    Identify and return an order ID in the format 'O####' (e.g., O2043).
    Returns None if not found.
    """
    m = re.search(r"\b(O\d{4})\b", text, flags=re.I)
    return m.group(1).upper() if m else None

# ------------------------------------------------------------
# STEP 4: Utility — Detect complaint intent from user input
# ------------------------------------------------------------
def is_complaint_intent(text: str):
  """
  Perform simple keyword-based complaint detection.
  Helps route angry or urgent customer queries to support.
  """
  complaint_keywords = [
      "complain", "resolution", "not received", "not recieved",
      "didn't receive", "angry", "bad service",
      "immediate response", "immediate"
    ]
  t = text.lower()
  return any(k in t for k in complaint_keywords)

# ------------------------------------------------------------
# STEP 5: Safety — Remove confidential columns before query
# ------------------------------------------------------------
def remove_confidential_columns(columns: list):
    """
    Sanitize SELECT clause by removing confidential columns
    (like timestamps or internal tracking fields) from results.
    """
    return [c for c in columns if c not in confidential_columns]

# ================================================================
#  SECTION 1: LLM Factory (Groq-only)
# ---------------------------------------------------------------
#  Builds a deterministic ChatGroq model for SQL reasoning.
#  Determinism ensures consistent, repeatable query responses.
# ================================================================
def _make_deterministic_llm():
    """
    Build a low-temperature ChatGroq LLM for SQL reasoning.
    Deterministic responses preferred for reproducible database answers.
    """
    # STEP 1: Check for presence of GROQ_API_KEY in environment
    # groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    # STEP 2: Instantiate ChatGroq LLM with very low temperature
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=groq_api_key,
        temperature=0.05,
    )


# ================================================================
#  SECTION 2: SQLite URI Helpers
# ---------------------------------------------------------------
#  Utility functions for resolving database paths and ensuring
#  correct URI formats for SQLite connections.
# ================================================================
def _resolve_sqlite_uri() -> str:
    """Resolve database URI path either from environment or default."""
    # STEP 1: Get DB path from environment (fallback: local file)
    db_path = os.getenv("DATA_DB_PATH", "customer_orders.db")
    #print('db_path = ', db_path)

    # STEP 2: Convert to SQLite URI format if needed
    return db_path if db_path.startswith("sqlite:///") else f"sqlite:///{db_path}"


def _resolve_path_from_uri(uri: str) -> str:
    """Extract filesystem path from a sqlite:/// URI."""
    # STEP 1: Handle different SQLite URI prefixes gracefully
    if uri.startswith("sqlite:///"):
        return uri.replace("sqlite:///", "", 1)
    if uri.startswith("sqlite://"):
        return uri.replace("sqlite://", "", 1)
    return uri


# ================================================================
#  SECTION 3: Schema Inspection and Caching
# ---------------------------------------------------------------
#  Reads database metadata such as table names, column definitions,
#  row counts, and sample rows. Also caches schema for quick reuse.
# ================================================================
def _inspect_schema(sqlite_uri: str, preview_rows: int = 3) -> Dict[str, Any]:
    """Return structural summary of all non-system tables."""
    # STEP 1: Resolve SQLite file path and establish connection
    db_path = _resolve_path_from_uri(sqlite_uri)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # STEP 2: Retrieve user-defined tables only (skip system tables)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r[0] for r in cur.fetchall()]
    schema: Dict[str, Any] = {}

    # STEP 3: Collect details for each table
    for table in tables:
        # 3a. Get column structure
        cur.execute(f"PRAGMA table_info({table});")
        cols = [
            {"id": r[0], "name": r[1], "type": r[2], "notnull": r[3], "default": r[4], "pk": r[5]}
            for r in cur.fetchall()
        ]

        # 3b. Get total row count
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            count = cur.fetchone()[0]
        except sqlite3.Error:
            count = None

        # 3c. Get sample preview rows
        try:
            cur.execute(f"SELECT * FROM {table} LIMIT {preview_rows};")
            sample = cur.fetchall()
        except sqlite3.Error:
            sample = []

        schema[table] = {"columns": cols, "row_count": count, "sample": sample}

    # STEP 4: Close DB connection
    conn.close()
    return schema


def _schema_summary(schema: Dict[str, Any], max_lines: int = 20) -> str:
    """Compact text summary for embedding in LLM system prompts."""
    # STEP 1: Prepare summary lines for each table
    lines: List[str] = []
    for table, meta in schema.items():
        lines.append(f"Table: {table} (rows: {meta.get('row_count')})")
        for col in meta.get("columns", [])[:max_lines]:
            pk = " PRIMARY_KEY" if col.get("pk") else ""
            lines.append(f"  - {col.get('name')} {col.get('type')}{pk}")
        if len(meta.get("columns", [])) > max_lines:
            lines.append(f"  - ... ({len(meta.get('columns', []))} columns total)")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _cached_schema(sqlite_uri: str, preview_rows: int = 3) -> Tuple[Dict[str, Any], str]:
    """Cache schema and textual summary to avoid repeated inspections."""
    # STEP 1: Fetch schema metadata and summarize
    schema = _inspect_schema(sqlite_uri, preview_rows)
    summary = _schema_summary(schema)
    return schema, summary


# ================================================================
#  SECTION 4: SQL Agent Construction
# ---------------------------------------------------------------
#  Creates the SQL agent object with system prompts and schema
#  embedded for contextual LLM reasoning. Enforces SELECT-only mode.
# ================================================================
def _build_sql_agent(
    db_uri: str,
    llm=None,
    preview_rows: int = 3,
    safe_limit: int = 100,
):
    """
    Create SQL agent with schema embedding and strict rule context.
    """
    # STEP 1: Initialize deterministic LLM if not provided
    llm = llm or _make_deterministic_llm()

    # STEP 2: Load or cache database schema
    schema, schema_text = _cached_schema(db_uri, preview_rows)

    # STEP 3: Prepare strict system prompt for LLM
    sys_prompt = textwrap.dedent(f"""
    You are FoodHub’s SQL assistant with read-only access to a customer database.
    MANDATORY CONSTRAINTS:
    1. Run only verified SQL queries — never assume unseen data.
    2. STRICTLY SELECT-only. No INSERT, UPDATE, DELETE, or schema modification.
    3. For unfiltered requests, cap results with LIMIT {safe_limit} and mention it in response.
    4. Include the executed SQL and a one-line plain English interpretation.
    5. If unsure, infer column meaning using the schema summary below.
    SCHEMA SNAPSHOT:
    {schema_text}
    """)

    # STEP 4: Create SQL agent using LangChain components
    db = SQLDatabase.from_uri(db_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(llm=llm, toolkit=toolkit, system_message=sys_prompt, verbose=False, top_k=100)

    return agent, schema, schema_text

# ================================================================
#  SECTION 5: Policy Filters and Safety Patterns
# ---------------------------------------------------------------
#  Regex filters to identify restricted query intents such as:
#  - Enumeration of all records
#  - Destructive (write) operations
#  - Refund/cancellation requests for human escalation
# ================================================================
_NEGATE_PATTERNS = [
    r"\bdoesn't\b", r"\bdoes not\b", r"\bnot\b", r"\bdo not\b",
    r"\bavoid\b", r"\bdon't\b"
]
_ENUM_PATTERNS = [
    r"\ball\b", r"\bevery\b", r"\bentire\b", r"\bcomplete\b", r"\bselect\s+\*\b",
    r"\bjoin\b", r"\bunion\b", r"\bcross\b", r"\bdatabase\b", r"\bpragma\b"
]
_DESTRUCT_PATTERNS = [
    r"\bdelete\b", r"\bmodify\b", r"\binsert\b", r"\bcreate\b", r"\balter\b",     # remove 'update'
    r"\bdrop\b", r"\btruncate\b", r"\battach\b", r"\bdetach\b"
]
_HUMAN_HANDOFF_PATTERNS = [r"\brefund\b", r"\bcancel\b", r"\breturn\b"]

def _matches_any(patterns: List[str], text: str) -> bool:
    """Check if user query matches any restricted pattern."""
    # STEP 1: Convert text to lowercase and compare with known patterns
    query = (text or "").lower()
    return any(re.search(p, query) for p in patterns)

# ================================================================
#  _query_id_match
# ---------------------------------------------------------------
#  Check if the query contains any customer ID and if that matches with
#  locked in customer ID.
#  if True, proceed further to process; else, retrun False.
# ================================================================
def _query_id_match(db_uri: str, cust_id: str, query: str) -> bool:
    """Verify that cust_id exists in at least one expected table."""
    # STEP 1: Resolve file path and connect to SQLite
    db_path = _resolve_path_from_uri(db_uri)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Step 2: Run SQL directly using the connection
    qc = f"SELECT order_id FROM orders WHERE cust_id='{cust_id}';"
    db_order_id = pd.read_sql_query(qc, conn)

    # STEP 3:
    # Extract customer ID if present in the query
    #hello = 0
    return_value = True
    qc_cid = []
    cidcnt = 0
    for match in re.findall(r"\bC\d{3,6}\b", query, flags=re.IGNORECASE):
        if match:
            cidcnt += 1
            qc_cid = match.upper()
            print('qc_cid = ', qc_cid)
            if qc_cid != cust_id:
                return_value = False

    # Extract order ID if present in the query
    qc_oid = []
    oidcnt = 0
    for match in re.findall(r"\bO\d{3,6}\b", query, flags=re.IGNORECASE):
        if match:
            oidcnt += 1
            qc_oid = match.upper()
            if qc_oid != db_order_id:
                return_value = False

    if qc_oid == [] and qc_cid == [] and return_value == True:
        #hello = 5
        return_value = True

    if oidcnt > 1 or cidcnt > 1:
        #hello = 6
        return_value = False

    #print('hello = ', hello)
    print('return_value = ', return_value)
    #print('qc_cid = ', qc_cid)
    #print('qc_oid = ', qc_oid)
    #print('db_order_id = ', db_order_id)
    #print('cust_id = ', cust_id)
    #print('query = ', query)

    # STEP 4: Close connection if not found
    conn.close()
    return return_value


# ================================================================
#  SECTION 6: Main Query Executor
# ---------------------------------------------------------------
#  Handles incoming user requests securely by:
#  1. Enforcing safety policies
#  2. Restricting scope to given cust_id
#  3. Invoking LLM for SQL generation and interpretation
# ================================================================
def order_query_tool_func(orderagent_input: str) -> str:
    """
    Accepts a stringified dict input like:
    "{'cust_id': 'C1018', 'user_query': 'What is the status of my order?'}"
    Enforces cust_id scope, safe syntax, and deterministic LLM logic.
    Parses it, authenticates, fetches data, and returns structured info.
    """
    try:
        # Safely parse the input string into a Python dictionary
        data = ast.literal_eval(orderagent_input)

        # Extract customer ID and user message from the parsed data
        cust_id = data.get("cust_id")
        user_query = data.get("user_query")
    except Exception:
        # If parsing fails, return a formatted error message
        # Return a stringified dictionary containing customer ID, orig_query, and db_orders
        return str({
            "cust_id": None,
            "orig_query": None,
            "db_orders": "⚠️ Invalid input format for OrderQueryTool."
        })

    try:
        # STEP 1: Policy-level checks — detect sensitive or restricted intents
        if not _matches_any(_NEGATE_PATTERNS, user_query):
        # For queries containing negative patterns, use LLM's decisions.
            if _matches_any(_HUMAN_HANDOFF_PATTERNS, user_query):
                # Return a stringified dictionary containing customer ID, orig_query, and db_orders
                return str({
                    "cust_id": cust_id,
                    "orig_query": user_query,
                    "db_orders": "I’ve sent your refund or cancellation request to our human support team. They’ll verify it and update you soon."
                })

            if _matches_any(_DESTRUCT_PATTERNS, user_query):
                # Return a stringified dictionary containing customer ID, orig_query, and db_orders
                return str({
                    "cust_id": cust_id,
                    "orig_query": user_query,
                    "db_orders": "Destructive database actions aren’t permitted. I can connect you to a human agent if you’d like help with changes."
                })

            if _matches_any(_ENUM_PATTERNS, user_query):
                # Return a stringified dictionary containing customer ID, orig_query, and db_orders
                return str({
                    "cust_id": cust_id,
                    "orig_query": user_query,
                    "db_orders": "For security, I can’t display full database contents or every customer’s data. Please ask about your own order or account instead."
                })

        # STEP 2: Initialize SQL agent and deterministic model
        db_uri = _resolve_sqlite_uri()
        llm = _make_deterministic_llm()
        agent, schema, schema_text = _build_sql_agent(db_uri, llm)

        # STEP 3: Validate if customer identity is provided in the query will match the id of locked in customer.
        if not _query_id_match(db_uri, cust_id, user_query):
            # Return a stringified dictionary containing customer ID, orig_query, and db_orders
            return str({
                "cust_id": cust_id,
                "orig_query": user_query,
                "db_orders": "Sorry, I cannot share records pertaining to another customer for privacy reasons. Please recheck your account details or reach support for assistance."
            })

        # STEP 4: Guarded prompt preparation for safe LLM execution
        guarded_prompt = textwrap.dedent(f"""
        {user_query}
        HARD REQUIREMENTS:
        - Restrict all SQL to cust_id = '{cust_id}'.
        - Perform SELECT-only operations.
        - Avoid exposing table structures or entire datasets.
        - Use LIMIT for large sets and note it in explanation.
        - Respond clearly with a short natural summary and the executed SQL.
        - Never exceed 15 seconds of reasoning time.
        """)

        # STEP 5: Execute SQL agent safely
        output = agent.invoke({"input": guarded_prompt})
        message = output.get("output", str(output)) if isinstance(output, dict) else str(output)

        # STEP 6: Handle empty or null responses gracefully
        if not message or message.strip().lower() in {"none", "null", "no results", "[]"}:
            # Return a stringified dictionary containing customer ID, orig_query, and db_orders
            return str({
                "cust_id": cust_id,
                "orig_query": user_query,
                "db_orders": "Sorry, I couldn’t find any data matching your request."
            })

        # Return a stringified dictionary containing customer ID, orig_query, and db_orders
        return str({
            "cust_id": cust_id,
            "orig_query": user_query,
            "db_orders": message
        })

    except Exception:
        # STEP 7: Catch and return formatted traceback for debugging
        message = f"Query execution error.\n```\n{traceback.format_exc()}\n```"
        return str({
            "cust_id": None,
            "orig_query": None,
            "db_orders": message
        })


# ================================================================
#  SECTION 7: LangChain Tool Wrapper
# ---------------------------------------------------------------
#  Wraps the SQL query executor as a callable Tool.
#  Enables integration with agent workflows that need database access.
# ================================================================
#from langchain.tools import Tool

#OrderQueryTool = Tool(
#    name="order_query_tool",
#    func=order_query_tool_func,
#    description="Use this tool to fetch order-related (read-only) info for a customer's order. Requires customer id from session. Blocks confidential fields. Returns structured output as a stringified dictionary"
#)
