import logging
from typing import Optional
import streamlit as st
from query_classifier import train_classifier, classify_query
from web_scraper import scrape
from llm_processor import get_static_response, process_dynamic_response
from db_handler import generate_sql_query, execute_sql_query

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_SCHEMA = """
Tables:
users (id INT, name VARCHAR, email VARCHAR)
products (id INT, name VARCHAR, price DECIMAL)
orders (id INT, user_id INT, product_id INT, quantity INT)
"""

def handle_query(query: str, model) -> str:
    """
    Handle user query based on its type (static, dynamic, sql).

    Args:
        query (str): User query.
        model: Trained classifier model.

    Returns:
        str: Response to the query.
    """
    query_type = classify_query(query, model)
    logger.info(f"Handling query: {query} as {query_type}")

    if query_type == "sql":
        sql = generate_sql_query(query, DB_SCHEMA)
        logger.info(f"Generated SQL: {sql}")
        return str(execute_sql_query(sql))
    elif query_type == "dynamic":
        results, live_info, snippet = scrape(query)
        return process_dynamic_response(query, results, live_info, snippet)
    else:
        return get_static_response(query)

def main() -> None:
    """Run the Streamlit application."""
    st.title("Davinators Query Bot :)")
    st.write("Dare to ask me anything")

    model = train_classifier()
    user_query = st.text_input("Enter your question:", placeholder="e.g., How can I be a billionaire?")
    if st.button("Send"):
        if user_query:
            with st.spinner("Processing your query..."):
                response = handle_query(user_query, model)
            st.success("Here is your response:")
            st.write(response)
        else:
            st.error("Please enter a query!")

if __name__ == "__main__":
    main()