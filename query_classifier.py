import logging
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Full training data as provided
TRAINING_DATA: List[Tuple[str, str]] = [
    ("What is the weather today?", "dynamic"),
    ("Tell me a joke", "static"),
    ("Latest news updates", "dynamic"),
    ("Rate of any currency", "dynamic"),
    ("tell me the weather", "dynamic"),
    ("Do you know", "dynamic"),
    ("USD rate in INR", "dynamic"),
    ("Current rate of USD", "dynamic"),
    ("How do I bake a cake?", "static"),
    ("What is langchain?", "static"),
    ("What is noun?", "static"),
    ("Current stock prices", "dynamic"),
    ("Today's news headlines", "dynamic"),
    ("Thank you or appreciation or anger", "static"),
    ("Hi, how are you?", "static"),
    ("I am happy", "static"),
    ("You are best", "static"),
    ("What are the latest updates in AI research?", "dynamic"),
    ("What’s the newest technology in AI?", "dynamic"),
    ("Tell me about the new developments in AI.", "dynamic"),
    ("What’s the best way to make money?", "static"),
    ("How can I earn more?", "static"),
    ("What are some tips for making money?", "static"),
    ("What is the forecast for the weather in New York?", "dynamic"),
    ("How much is Bitcoin right now?", "dynamic"),
    ("What happened in the last football game?", "dynamic"),
    ("What was the price of Bitcoin in 2017?", "static"),
    ("What is the exchange rate between USD and EUR?", "dynamic"),
    ("What is the current USD to EUR exchange rate?", "dynamic"),
    ("What is the price of Apple stock?", "dynamic"),
    ("Give me a recipe for a cake.", "static"),
    ("Teach me how to bake a cake.", "static"),
    ("What is the current time in London?", "dynamic"),
    ("What is the price of gold today?", "dynamic"),
    ("What is the price of crude oil?", "dynamic"),
    ("What is the capital of Japan?", "static"),
    ("Who won the latest election?", "dynamic"),
    ("When is the next big sporting event?", "dynamic"),
    ("What are the latest trends in technology?", "dynamic"),
    ("What is the best programming language?", "static"),
    ("Can you tell me the current temperature in Berlin?", "dynamic"),
    ("What’s the best strategy to make money from real estate?", "static"),
    ("What’s the most popular music genre right now?", "dynamic"),
    ("How do I use Langchain in a Python project?", "static"),
    ("What’s the best way to learn machine learning?", "static"),
    ("What is the price of Bitcoin in 2023?", "static"),
    ("What happened in the latest stock market crash?", "dynamic"),
    ("How much did Apple earn last quarter?", "dynamic"),
    ("Can you suggest some stock investment tips?", "static"),
    ("What are the stock prices for Tesla today?", "dynamic"),
    ("How do I build a website?", "static"),
    ("What are the best websites for learning Python?", "static"),
    ("How is the economy doing today?", "dynamic"),
    ("What are the top tech companies to watch?", "dynamic"),
    ("How much does a Bitcoin cost today?", "dynamic"),
    ("What’s the weather in San Francisco tomorrow?", "dynamic"),
    ("What’s the average income of a software developer?", "static"),
    ("When does the stock market open?", "dynamic"),
    ("What time does the stock market close?", "dynamic"),
    ("What is the capital of Canada?", "static"),
    ("What was the temperature in New York last week?", "static"),
    ("How do I bake a pie?", "static"),
    ("Give me a stock forecast for next week.", "dynamic"),
    ("What are the best practices for AI development?", "static"),
    ("What are the top 5 richest countries in the world?", "static"),
    ("What’s the best investment for 2025?", "dynamic"),
    ("What’s the latest news about the electric car industry?", "dynamic"),
    ("What was the price of gold in 2005?", "static"),
    ("Give me some stock trading tips.", "static"),
    ("What was the latest price of Bitcoin?", "dynamic"),
    ("Tell me about the latest trends in e-commerce.", "dynamic"),
    ("What are the best tips for self-development?", "static"),
    ("Can you suggest some great books on economics?", "static"),
    ("What is the exchange rate between GBP and JPY?", "dynamic"),
    ("Can you tell me the current exchange rate for USD?", "dynamic"),
    ("What is the next big cryptocurrency?", "dynamic"),
    ("What is the weather like in Tokyo today?", "dynamic"),
    ("How much is the price of silver today?", "dynamic"),
    ("When is the next World Cup?", "dynamic"),
    ("What’s the price of real estate in New York?", "dynamic"),
    ("How much is the price of crude oil today?", "dynamic"),
    ("What is the best cryptocurrency to invest in right now?", "dynamic"),
    ("What was the most significant tech advancement in 2024?", "dynamic"),
    ("What is the current exchange rate of USD to CNY?", "dynamic"),
    ("What is your name?", "static"),
    ("How old are you?", "static"),
    ("Where are you from?", "static"),
    ("Tell me about your family", "static"),
    ("What is your favorite color?", "static"),
    ("What are your hobbies?", "static"),
    ("What did you have for lunch?", "static"),
    ("How do I tie my shoes?", "static"),
    ("Do you like movies?", "static"),
    ("How do I get to the nearest store?", "static"),
    ("Can you recommend a good book?", "static"),
    ("How do I improve my fitness?", "static"),
    ("How can I make friends?", "static"),
    ("What is the best way to relax?", "static"),
    ("What is your favorite food?", "static"),
    ("How do I make tea?", "static"),
    ("How do I clean my house?", "static"),
    ("What is the meaning of life?", "static"),
    ("What are your favorite songs?", "static"),
    ("How do I start learning a new language?", "static"),
    ("What is your favorite movie?", "static"),
    ("How do I organize my day?", "static"),
    ("What is the capital of France?", "static"),
    ("How do I grow my own vegetables?", "static"),
    ("Show me all the users of the database", "sql"),
    ("Insert a new user with name Alice and email alice@example.com into the database", "sql"),
    ("Update the user with ID 2 to have the email alice_updated@example.com", "sql"),
    ("Delete user with ID 3", "sql"),
    ("List all products in the database", "sql"),
    ("Insert a new product with name 'Smartphone' and price 700", "sql"),
    ("Update product with ID 5 to have price 750", "sql"),
    ("Delete product with ID 4", "sql"),
    ("Show all orders in the database", "sql"),
    ("Add a new order for user ID 1 with product ID 2 and quantity 3", "sql"),
    ("Update order with ID 10 to set quantity to 5", "sql"),
    ("Delete order with ID 12", "sql"),
    ("Show the total number of users", "sql"),
    ("Show the total number of products", "sql"),
    ("List all orders placed by user with ID 2", "sql"),
    ("Find all products that cost more than 500", "sql"),
    ("Show all users who have placed an order", "sql"),
    ("Find the order with the highest quantity", "sql"),
    ("List all users who registered after 2022", "sql"),
    ("Show the average price of all products", "sql"),
    ("Find all orders that were placed on 2023-01-15", "sql"),
    ("Insert a new user with name John and email john@example.com of the database", "sql"),
    ("Update user with ID 1 to have email john_updated@example.com of the database", "sql"),
    ("Delete user with ID 2", "sql"),
    ("What are the latest stock prices?", "dynamic"),
    ("How much is Bitcoin right now?", "dynamic"),
    ("What are the top tech companies to watch?", "dynamic"),
    ("List all products", "sql"),
    ("Insert a new product with name 'Laptop' and price 1200", "sql"),
    ("Update product with ID 3 to have price 1300", "sql"),
    ("Delete product with ID 4", "sql"),
    ("Write a query to delete product with ID 4", "sql"),
    ("Write a query to see all the users", "sql"),
    ("Write a query to see all the products", "sql"),
    ("Write a query to update the age", "sql"),
    ("database", "sql")
]

def train_classifier() -> Pipeline:
    """
    Train a classifier to categorize queries as static, dynamic, or sql.

    Returns:
        Pipeline: Trained scikit-learn pipeline with TfidfVectorizer and MultinomialNB.
    """
    logger.info("Training query classifier")
    queries, labels = zip(*TRAINING_DATA)
    model: Pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(queries, labels)
    logger.info("Classifier trained successfully")
    return model

def classify_query(query: str, model: Pipeline) -> str:
    """
    Classify a query as 'static', 'dynamic', or 'sql'.

    Args:
        query (str): User input query.
        model (Pipeline): Trained classifier model.

    Returns:
        str: Query type ('static', 'dynamic', or 'sql').
    """
    logger.info(f"Classifying query: {query}")
    prediction: str = model.predict([query])[0]
    logger.info(f"Query classified as: {prediction}")
    return prediction

if __name__ == "__main__":
    model: Pipeline = train_classifier()
    test_query: str = "What is the weather today?"
    result: str = classify_query(test_query, model)
    logger.info(f"Test query result: {result}")