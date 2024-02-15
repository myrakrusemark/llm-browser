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

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Initialize Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
chrome_options.add_argument("--window-size=1920x1080")  # Set window size

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)


class BrowseWeb:

    def __init__(self):
        # Ensure necessary NLTK resources are downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        self.avg_token_length = 3
        self.max_tokens_returned = 5000
        self.max_output_length = self.max_tokens_returned * self.avg_token_length
        self.search_results_per_url = 2
        self.max_urls = 3
        self.batch_size = 5
        # Create a pipeline for feature extraction
        self.feature_extraction = pipeline('feature-extraction', model='bert-base-uncased')
        self.max_passage_token_length = 200
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def main(self, url, search_term=""):
        print("SEARCH TERM: " + search_term)
        combined_output = ""
        if not url.startswith("https://"):
            url = "https://" + url

        # Generate embedding for search term
        search_term_embedding = self.generate_embedding(search_term)
        if isinstance(search_term_embedding, list):
            search_term_embedding = torch.Tensor(search_term_embedding)

        # Retrieve webpage content and strip HTML tags
        passages, full_text = self.get_webpage_content(url)
        if not passages:
            print("UNABLE TO RETRIEVE PAGE CONTENT")
            output = "------------------------------------------------------\n"
            output += "Error: Unable to retrieve webpage content for " + url + "\n"
            output += "------------------------------------------------------\n\n"
            combined_output += output
        elif len(full_text) <= self.max_output_length:
            print("Using full page content")
            output = "------------------------------------------------------\n"
            output += "Full text for " + url + "\n"
            output += "------------------------------------------------------\n\n"
            output += full_text + "\n\n"
            combined_output += output
        else:
            # Compare each passage to the search term and order by relevance
            num_passages = len(passages)
            results = []
            print("Searching " + str(num_passages) + " passages for relevant information...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                passage_to_future = {}
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
            output += "Relevant passages for " + url + "\n"
            output += "------------------------------------------------------\n\n"
            for result in results[:self.search_results_per_url]:
                index = result["index"]
                passages_to_include = []
                
                # Include up to three previous passages if they exist
                for i in range(max(0, index - 3), index):
                    prev_passage = passages[i][1]  # Get the text of the previous passage
                    passages_to_include.append(prev_passage)
                
                # Include the current (matching) passage
                passages_to_include.append(result["passage"])
                
                # Include up to three next passages if they exist
                for i in range(index + 1, min(len(passages), index + 4)):
                    next_passage = passages[i][1]  # Get the text of the next passage
                    passages_to_include.append(next_passage)
                
                # Append the passages to the output
                for passage in passages_to_include:
                    output += passage + "\n\n"
            
            combined_output += output

        return combined_output


    def get_passage_result(self, passages, search_term_embedding):
        results = []
        for index, passage in passages:
            # Generate embedding for each passage individually
            passage_embedding = self.generate_embedding(passage)
            if isinstance(passage_embedding, list):
                passage_embedding = torch.Tensor(passage_embedding)
            score = torch.cosine_similarity(search_term_embedding, passage_embedding, dim=0)
            score = score.mean().item()
            results.append({'passage': passage, 'confidence': score, 'index': index})
        return results


    def generate_embedding(self, text):
        # Ensure text is a string
        if not isinstance(text, str):
            print(text)
            raise ValueError("Input text must be of type str.")
        
        # Directly pass the text to the feature_extraction pipeline
        with torch.no_grad():
            embeddings = self.feature_extraction(text)
        
        # Convert embeddings to a tensor and calculate the mean across the sequence dimension
        embedding_tensor = torch.tensor(embeddings)
        mean_embedding = embedding_tensor.mean(dim=1).squeeze().tolist()
        
        return mean_embedding



    def get_webpage_content(self, url):
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            full_text = driver.find_element(By.TAG_NAME, "body").text
            passage_length = self.max_passage_token_length * self.avg_token_length
            sections = [(int(i/passage_length), self.preprocess_text(full_text[i:i+passage_length])) for i in range(0, len(full_text), passage_length)]
            return sections, self.preprocess_text(full_text)
        except Exception as e:
            return False, False
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs, symbols, and numbers
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        # Rejoin tokens into a string
        return ' '.join(stemmed_tokens)

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
    #parser = argparse.ArgumentParser(description='Web page content extractor.')
    #parser.add_argument('url', type=str, help='The URL of the web page to extract content from.')
    #parser.add_argument('--request', type=str, default="", help='Optional search request for filtering content.')
    #args = parser.parse_args()
    url = "https://www.cnn.com"
    request = "gaza"

    browser = BrowseWeb()
    #output = browser.main([args.url], args.request)
    output = browser.main(url, request)

    print(output)
