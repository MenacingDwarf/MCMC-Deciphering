import re
import string
import random
import math

import numpy as np
from collections import Counter
from corus import load_wiki

class MCMCDeciphrator:
    def __init__(self, chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя '):
        self.chars_list = list(chars)
        simple_chars_dict = dict()
        for ch in self.chars_list:
            simple_chars_dict[ch] = 0

        self.chars_dict = dict()
        for ch in self.chars_list:
            self.chars_dict[ch] = simple_chars_dict.copy()
    
    #---------------------------------
    #--- Corpus collection methods ---
    #---------------------------------
        
    def clear_text(self, text):
        text = text.lower()
        text=" ".join([re.sub('[^а-яё]', '', ch) for ch in text.split()])
        text = re.sub(r'\s+', ' ', text)
        return text
        
    def count_dist(self, text, counter):
        counter += Counter(text)
        for i in range(len(text)-1):
            self.chars_dict[text[i]][text[i+1]] += 1
        return counter
        
    def get_dist(self, n_steps):
        counter = Counter()
        for i in range(n_steps):
            text = self.clear_text(next(self.records).text)
            counter = self.count_dist(text, counter)
        return counter

    def normal_dict(self):
        for key, value in self.chars_dict.items():
            all_sum = sum(value.values())
            if all_sum!=0:
                for k, val in value.items():
                    value[k] = val/all_sum
                    
    def collect_corpus(self, corus_path):
        self.records = load_wiki(corus_path)
        self.counter = self.get_dist(100000)
        self.normal_dict()                    
    
    #---------------------------------
    #---- MCMC deciphering methods ---
    #---------------------------------
    
    def get_message_probability(self, message):
        prob = 0
        for i in range(len(message)-1):
            if self.chars_dict[message[i]][message[i+1]]!=0:
                prob+= math.log(self.chars_dict[message[i]][message[i+1]])
        return prob
        
    def replace_char(self, mes, ch, to_ch):
        mes = re.sub(to_ch, 'А', mes)
        mes = re.sub(ch, to_ch, mes)
        mes = re.sub('А', ch, mes)
        return mes
        
    def decipher_message(self, message, n_steps):
        it = 0
        max_prob = self.get_message_probability(message)
        best_mes = message
        
        # main loop
        for i in range(n_steps):
            old_prob = self.get_message_probability(message)
            
            # random replace character for new message
            сh = random.sample(list(set(message)), 1)[0]
            to_ch = random.sample(self.chars_list, 1)[0]
            new_message = self.replace_char(message, сh, to_ch)
            new_prob = self.get_message_probability(new_message)
            
            # check if new message gives better result
            if random.uniform(0, 1) < math.exp(new_prob - old_prob):
                it = i
                message = new_message
                if new_prob > max_prob:
                    max_prob = new_prob
                    best_mes = message
                    
        return best_mes, max_prob, it
