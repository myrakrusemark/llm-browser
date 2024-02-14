import argparse
import logging
from bs4 import BeautifulSoup
import sys
from threading import Thread
import itertools
import time
import json
import concurrent.futures
import shutil
import torch
from transformers import pipeline, BertTokenizer
import re
import math
import itertools

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
#from selenium.webdriver.chrome.options import Options
#from selenium.webdriver.common.by import By
#from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support import expected_conditions as EC
#from webdriver_manager.chrome import ChromeDriverManager


# Set up Chrome options for running in headless mode
#chrome_options = Options()
#chrome_options.add_argument("--headless")  # Ensure GUI is off

#chrome_options.add_argument("--log-level=3")


class BrowseWeb:

    def __init__(self):
        self.avg_token_length = 3
        self.max_tokens_returned = 5000
        self.max_output_length = self.max_tokens_returned * self.avg_token_length
        self.search_results_per_url = 2
        self.max_urls = 3
        self.batch_size = 5
        # Create a pipeline for feature extraction
        self.feature_extraction = pipeline('feature-extraction', model='bert-base-uncased')
        self.max_passage_token_length = 500
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Update the Service to use ChromeDriverManager
        self.service = Service(ChromeDriverManager().install())
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument('--disable-cache')
        # Update other options as required...
        self.driver = webdriver.Chrome(service=self.service, options=self.options)

    def main(self, urls, search_term=""):
        print("SEARCH TERM: "+search_term)
        combined_output = ""
        for url in urls[:self.max_urls]:
            if not url.startswith("https://"):
                url = "https://" + url

            # Generate embedding for search term
            search_term_embedding = self.generate_embedding(search_term)
            if isinstance(search_term_embedding, list):
                search_term_embedding = torch.Tensor(search_term_embedding)

            # Retrieve webpage content and strip HTML tags
            #tion = LoadingAnimation()
            #animation.start(url)
            passages, full_text = self.get_webpage_content(url)
            if not passages:
                print("UNABLE TO RETRIEVE PAGE CONTENT")
                output = "------------------------------------------------------\n"
                output += "Error: Unable to retrieve webpage content for "+url+"\n"
                output += "------------------------------------------------------\n\n"
                combined_output += output
            # If the page is small, then just give the full text
            elif len(full_text) <= self.max_output_length:
                print("Using full page content")
                output = "------------------------------------------------------\n"
                output += "Full text for "+url+"\n"
                output += "------------------------------------------------------\n\n"
                output += full_text+"\n\n"
                combined_output += output
            else:
                # Compare each passage to the search term and order by relevance
                num_passages = len(passages)
                results = []
                print("Searching " + str(num_passages) + " passages for relevant information...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Batch the passages
                    passage_to_future = {}  # Map from Future to batch of passages
                    for i in range(0, len(passages), self.batch_size):
                        batch = passages[i:i + self.batch_size]
                        future = executor.submit(self.get_passage_result, batch, search_term_embedding)
                        passage_to_future[future] = batch

                    for future in concurrent.futures.as_completed(passage_to_future):
                        results.extend(future.result())

                # Sort results by confidence
                results.sort(key=lambda x: x['confidence'], reverse=True)

                # Generate output
                output = "------------------------------------------------------\n"
                output += "Relevant passages for "+url+"\n"
                output += "------------------------------------------------------\n\n"
                for result in results[:self.search_results_per_url]:
                    output += result["passage"] + "\n\n"
                combined_output += output

        # Return truncated, ordered, results
        return combined_output

    def get_passage_result(self, passages, search_term_embedding):
        passage_embeddings = self.generate_embedding([p[1] for p in passages])
        results = []
        for (index, passage), passage_embedding in zip(passages, passage_embeddings):
            score = torch.cosine_similarity(search_term_embedding, torch.Tensor(passage_embedding), dim=0)
            score = score.mean().item()
            results.append({'passage': passage, 'confidence': score, 'index': index})
        return results

    def generate_embedding(self, text):
        # Ensure text is a string
        if not isinstance(text, str):
            raise ValueError("Input text must be of type str.")
        
        # Directly pass the text to the feature_extraction pipeline
        with torch.no_grad():
            embeddings = self.feature_extraction(text)
        
        # Convert embeddings to a tensor and calculate the mean across the sequence dimension
        embedding_tensor = torch.tensor(embeddings)
        mean_embedding = embedding_tensor.mean(dim=1).squeeze().tolist()
        
        return mean_embedding



    def get_webpage_content(self, url):
        # Set up the Chrome driver
        try:
            # Load the webpage
            self.driver.get(url)
            # Wait for the page to fully load
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            # Get the page source
            page_source = self.driver.page_source
            # Load page source into BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            # Get full text
            full_text = soup.get_text()
            # Remove duplicate newline characters
            full_text = re.sub('\n+', '\n', full_text)
            passage_length = self.max_passage_token_length * self.avg_token_length
            # Split tokens into sections
            sections = [(int(i/passage_length), full_text[i:i+passage_length]) for i in range(0, len(full_text), passage_length)]
            return sections, full_text
        except Exception as e:
            return False, False
        finally:
            # Stop the loading animation
            #animation.stop()
            # Close the WebDriver
            #driver.quit()
            pass

class LoadingAnimation:
    def __init__(self):
        self.stop_loading = False

    def start(self, text):
        self.animation = Thread(target=self.animate, args=(text,))
        self.animation.start()

    def animate(self, text):
        for c in itertools.cycle(['.  ', '.. ', '...', '   ']):
            if self.stop_loading:
                break
            sys.stdout.write('\rLoading ' + text + c)
            sys.stdout.flush()
            time.sleep(0.5)
        sys.stdout.write('\rDone!        ')

    def stop(self):
        self.stop_loading = True
        self.animation.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Web page content extractor.')
    parser.add_argument('url', type=str, help='The URL of the web page to extract content from.')
    parser.add_argument('--request', type=str, default="", help='Optional search request for filtering content.')
    args = parser.parse_args()

    browser = BrowseWeb()
    output = browser.main([args.url], args.request)
    print(output)
