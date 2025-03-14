import logging
from typing import Union, List, Dict
import mysql.connector
from langchain.llms import Ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "datascience",
    "database": "sample_db",
}

def generate_sql_query(query: str, db_schema: str) -> str:
    """
    Generate an SQL query from a natural language input using Ollama.

    Args:
        query (str): User query.
        db_schema (str): Database schema description.

    Returns:
        str: Generated SQL query.
    """
    logger.info(f"Generating SQL for query: {query}")
    llm = Ollama(model="llama3.2")
    prompt = f"""
    You are an AI trained to convert natural language into SQL queries.
    The database schema is as follows:
    {db_schema}
    Convert the following user query into SQL:
    "{query}"
    """
    response: str = llm.invoke(prompt)
    logger.info(f"Generated SQL: {response}")
    return response

def execute_sql_query(sql_query: str) -> Union[str, List[Dict]]:
    """
    Execute an SQL query on the MySQL database.

    Args:
        sql_query (str): SQL query to execute.

    Returns:
        Union[str, List[Dict]]: Query results or execution status.
    """
    logger.info(f"Executing SQL: {sql_query}")
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        cursor.execute(sql_query)

        if sql_query.strip().lower().startswith("select"):
            results: List[Dict] = cursor.fetchall()
        else:
            connection.commit()
            results = f"Query executed successfully: {sql_query}"

        cursor.close()
        connection.close()
        logger.info(f"SQL execution result: {results}")
        return results
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return f"Database error: {err}"

if __name__ == "__main__":
    schema = """
    Tables:
    users (id INT, name VARCHAR, email VARCHAR)
    products (id INT, name VARCHAR, price DECIMAL)
    orders (id INT, user_id INT, product_id INT, quantity INT)
    """
    sql = generate_sql_query("Show me all the users of the database", schema)
    result = execute_sql_query(sql)
    logger.info(f"Test result: {result}")