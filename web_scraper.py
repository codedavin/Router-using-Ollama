import logging
from typing import Tuple, List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def scrape(query: str) -> Tuple[List[Dict], Optional[str], Optional[str]]:
    """
    Scrape Google for dynamic query responses.

    Args:
        query (str): User query to search.

    Returns:
        Tuple[List[Dict], Optional[str], Optional[str]]: Scraped results, live info, and featured snippet.
    """
    logger.info(f"Starting web scrape for query: {query}")
    options = Options()
    # adding argument to disable the AutomationControlled flag 
    options.add_argument("--disable-blink-features=AutomationControlled") 
    
    # exclude the collection of enable-automation switches 
    options.add_experimental_option("excludeSwitches", ["enable-automation"]) 
    
    # turn-off userAutomationExtension 
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    results: List[Dict] = []
    live_info: Optional[str] = None
    featured_snippet: Optional[str] = None

    try:
        driver.get("https://www.google.com")
        logger.info("Opened Google homepage")

        search_box = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        logger.info(f"Searched for: {query}")

        WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.tF2Cxc"))
        )

        if "weather" in query.lower():
            try:
                weather_widget = driver.find_element(By.ID, "wob_wc")
                temperature = weather_widget.find_element(By.ID, "wob_tm").text
                weather_description = weather_widget.find_element(By.CSS_SELECTOR, "img").get_attribute("alt")
                live_info = f"Live Weather: {temperature}°C, {weather_description}"
                logger.info(f"Found live weather: {live_info}")
            except:
                logger.info("No live weather info found")

        result_elements = driver.find_elements(By.CSS_SELECTOR, "div.tF2Cxc")
        for element in result_elements:
            try:
                title = element.find_element(By.TAG_NAME, "h3").text
                url = element.find_element(By.TAG_NAME, "a").get_attribute("href")
                snippet = ""
                try:
                    snippet = element.find_element(By.CSS_SELECTOR, "div.VwiC3b.yXK7lf.p4wth.r025kc.hJNv6b.Hdw6tb").text
                except:
                    snippet = ""
                try:
                    date_element = element.find_element(By.CSS_SELECTOR, "span.f")
                    date_text = date_element.text
                    article_date = datetime.strptime(date_text, '%b %d, %Y')
                    if article_date > datetime.now() - timedelta(days=1):
                        results.append({"title": title, "url": url, "snippet": snippet, "date": article_date})
                except:
                    results.append({"title": title, "url": url, "snippet": snippet, "date": None})
            except Exception as e:
                logger.warning(f"Error processing result: {e}")

        try:
            featured_snippet_element = driver.find_element(By.CSS_SELECTOR, "div.VkpGBb")
            featured_snippet = featured_snippet_element.text
            logger.info(f"Found featured snippet: {featured_snippet}")
        except:
            logger.info("No featured snippet found")

        logger.info(f"Scraped {len(results)} results")
    except Exception as e:
        logger.error(f"Scraping error: {e}")
    finally:
        driver.quit()

    return results, live_info, featured_snippet

if __name__ == "__main__":
    results, live_info, snippet = scrape("What is the weather today?")
    logger.info(f"Results: {results[:2]}, Live Info: {live_info}, Snippet: {snippet}")