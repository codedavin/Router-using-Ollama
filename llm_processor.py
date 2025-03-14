import logging
from typing import Dict, List, Optional
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_static_response(query: str) -> str:
    """
    Generate a static response using LangChain and Ollama.

    Args:
        query (str): User query.

    Returns:
        str: Static response.
    """
    logger.info(f"Generating static response for: {query}")
    llm = Ollama(model="llama3.1")
    memory = ConversationBufferMemory()
    prompt_template = PromptTemplate(
        template="You are a polite chatbot. Provide a short and sweet answer to the following query:\n{user_query}",
        input_variables=["user_query"]
    )
    chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)
    response: str = chain.run({"user_query": query})
    logger.info(f"Static response: {response}")
    return response

def initialize_dynamic_chain() -> LLMChain:
    """
    Initialize LangChain for dynamic query processing.

    Returns:
        LLMChain: Configured chain for dynamic responses.
    """
    logger.info("Initializing dynamic LangChain")
    llm = Ollama(model="llama3.2")
    memory = ConversationBufferMemory()
    prompt_template = PromptTemplate(
        template="""
        You are a helpful chatbot. Your role is to give accurate, real-time answers based on the most recent data.
        Rules:
        1. Prioritize live data provided.
        2. Use the latest info from the data fed to you.
        3. Keep answers crisp, short, and accurate.
        User's question: {combined_input}
        Most recent data: {combined_input}
        """,
        input_variables=["combined_input"]
    )
    chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)
    return chain

def process_dynamic_response(query: str, results: List[Dict], live_info: Optional[str], snippet: Optional[str]) -> str:
    """
    Process dynamic query with scraped data.

    Args:
        query (str): User query.
        results (List[Dict]): Scraped search results.
        live_info (Optional[str]): Live data (e.g., weather).
        snippet (Optional[str]): Featured snippet.

    Returns:
        str: Dynamic response.
    """
    logger.info(f"Processing dynamic response for: {query}")
    if featured_snippet := snippet:
        return f"Here's what I found for you:\n{featured_snippet}"
    if live_info:
        return live_info
    if results:
        formatted_results = "\n".join(
            [f"Title: {result['title']}\nSnippet: {result['snippet']}\nURL: {result['url']}" for result in results]
        )
        combined_input = f"User Input: {query}\n\nMost Recent Data:\n{formatted_results}"
        chain = initialize_dynamic_chain()
        response: str = chain.run({"combined_input": combined_input})
        return response
    return "Sorry, I couldn't find any relevant live data."

if __name__ == "__main__":
    response = get_static_response("Tell me a joke")
    logger.info(f"Static test: {response}")