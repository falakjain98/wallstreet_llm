from typing import Optional
from transformers import AutoTokenizer
import re

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MIN_TOKENS = 10
MAX_TOKENS = 160
MIN_CHARS = 20
CEILING_CHARS = MAX_TOKENS * 7

class Item:
    """
    An Item is a cleaned, curated datapoint of a Tweet with a Sentiment Label
    """
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Sentiment is "
    QUESTION = "What is the financial sentiment of this tweet?"
    REMOVALS = [] #['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", "Package", ":", "Number of", "Best Sellers", "Number", "Product "]

    text: str
    label: float
    label_category: str
    prompt: Optional[str] = None
    include = False

    def __init__(self, data, label_category):
        self.text = data['text']
        self.label = data['label']
        self.label_category = label_category
        self.parse(data)

    def scrub_details(self):
        """
        Clean up the details string by removing common text that doesn't add value
        """
        text = self.text
        for remove in self.REMOVALS:
            text = text.replace(remove, "")
        return text

    def scrub(self, stuff):
        """
        Clean up the provided text by removing unnecessary characters and whitespace
        Also remove words that are 7+ chars and contain numbers, as these are likely irrelevant product numbers
        """
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = stuff.split(' ')
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    def parse(self, data):
        """
        Parse this datapoint and if it fits within the allowed Token range,
        then set include to True
        """
        contents = data['text']
        if contents:
            contents += '\n'
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(contents)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True

    def make_prompt(self, text):
        """
        Set the prompt instance variable to be a prompt appropriate for training
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(self.label_category)}"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Return a prompt suitable for testing, with the actual label removed
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        Return a String version of this Item
        """
        return f"<{self.text} = {self.label_category}>"

        

    
    