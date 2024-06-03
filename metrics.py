import os
import json
import torch
import argparse
import numpy as np
import string
import pickle
import dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
import gc
import time
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", type=str, default="")
parser.add_argument("--labels", type=str, default="/data2/home/yaoziqi/Projects/Learning/NLP/ESC_Chatbot/data/gpt2/ESC_test.pkl")
parser.add_argument("--references", type=str, default="/data2/home/yaoziqi/Projects/Learning/NLP/ESC_Chatbot/data/gpt2/ESC_test.pkl")
args = parser.parse_args()

labels_tokens = args.labels
references_tokens = args.references
with open(labels_tokens, "rb") as f:
    labels_tokens = pickle.load(f)
with open(references_tokens, "rb") as f:
    references_tokens = pickle.load(f)
# print (references_tokens['input_ids)
references = [
    ["The", "cat", "sat", "on", "the", "mat"],
    ["She", "sells", "seashells", "by", "the", "seashore"],
    ["A", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["The", "rain", "in", "Spain", "stays", "mainly", "in", "the", "plain"],
    ["To", "be", "or", "not", "to", "be", "that", "is", "the", "question"],
    ["All", "that", "glitters", "is", "not", "gold"],
    ["Actions", "speak", "louder", "than", "words"],
    ["The", "early", "bird", "catches", "the", "worm"],
    ["A", "picture", "is", "worth", "a", "thousand", "words"],
    ["When", "in", "Rome", "do", "as", "the", "Romans", "do"]
]
labels = [
    ["The", "cat", "slept", "on", "the", "mat"],
    ["She", "sells", "flowers", "by", "the", "seashore"],
    ["A", "quick", "brown", "fox", "leaps", "over", "the", "lazy", "dog"],
    ["The", "rain", "in", "Spain", "falls", "mainly", "on", "the", "plain"],
    ["To", "be", "or", "not", "to", "be", "that", "is", "my", "question"],
    ["All", "that", "glitters", "may", "not", "be", "gold"],
    ["Actions", "speak", "louder", "than", "promises"],
    ["The", "early", "bird", "gets", "the", "worm"],
    ["A", "picture", "is", "worth", "a", "thousand", "thoughts"],
    ["When", "in", "Rome", "act", "as", "the", "Romans", "do"]
]

def tokens_to_sentence(words):
    stripped_words = [word.strip() for word in words]
    sentence = ' '.join(stripped_words)
    return sentence


def calculate_rouge(references, labels):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_count = 0
    rouge_score = 0.0
    for reference, label in zip(references, labels):
        reference_sentence = tokens_to_sentence(reference)
        label_sentence = tokens_to_sentence(label)
        score = scorer.score(reference_sentence, label_sentence)
        rouge_score += score["rougeL"].fmeasure
        total_count +=1
    return rouge_score/total_count
def calculate_bleu(references, labels):
    total_count = 0
    bleu2_score = 0.0
    for reference, label in zip(references, labels):
        score = sentence_bleu([reference], label, (0.5,0.5))
        if(bleu2_score == 0.0):
            bleu2_score += score
        else:
            bleu2_score *= score
        total_count += 1
    return bleu2_score**(1 / total_count)
print(f"Rouge-L Score: {calculate_rouge(references, labels)}")
print(f"Bleu-2 Score: {calculate_bleu(references, labels)}") 