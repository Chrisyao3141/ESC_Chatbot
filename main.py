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
from transformers.utils import logging
import evaluate

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int, default=0)
parser.add_argument("--infer", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_path", type=str, default="/data/sam/Chris/gpt-2/gpt2")
parser.add_argument("--data_directory", type=str, default="./data/gpt2")
parser.add_argument("--num_epochs", type=int, default=4)
parser.add_argument("--device", type=int, default=-1)
parser.add_argument("--output_directory", type=str, default= "/data/sam/Chris/")
parser.add_argument("--checkpoint", type=str, default="")
args = parser.parse_args()

batch_size = args.batch_size
mode = args.train
infer = args.infer
num_epochs = args.num_epochs
model_path = args.model_path #"/data/sam/Chris/gpt-2/gpt2"
output_dir = args.output_directory
checkpoint = args.checkpoint
data_directory = args.data_directory
train_data_directory = os.path.join(data_directory, "ESC_train.pkl")
validation_data_directory = os.path.join(data_directory, "ESC_validation.pkl")
test_data_directory = os.path.join(data_directory, "ESC_test.pkl")
device = torch.device(f"cuda:{args.device}" if args.device >=0 else "cpu")

# print(loaded_data[0].tokens)
# print(train_data)
# print(test_data['input_ids'])
# print(len(train_dataset))
# print(len(test_dataset))

def train(model, optimizer):
    global_training_steps = 0
    if(checkpoint != ""):
        checkpoint = model_path.split("-")
        checkpoint = int(checkpoint[-1])
        global_training_steps+= checkpoint
    eval_loss_list = []
    epoch_count = 0
    epoch_loss_list = []
    train_data = []
    with open(train_data_directory, 'rb') as f:
        train_data = pickle.load(f)
        print("Training Data Loaded")
    train_dataset = dataset.ESCDataset(train_data)
    dataloader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    print("Beginning training")
    start = time.time()
    for epoch in range(num_epochs):
        epoch_steps = 0
        for batch in dataloader:
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            # if (global_training_steps % 1000 == 0):
            #     print(input_ids)
            attention_mask = batch['attention_mask'].to(device)
            # print(len(input_ids[0]))
            # print((attention_mask))
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            print(outputs)
            print(outputs.keys())
            loss = outputs[0]
            train_loss += loss.mean()
            loss.backward()
            optimizer.step()
            global_training_steps += 1
            epoch_steps +=1

            # basic progress logging
            if (global_training_steps % 1000 == 0):
                current = time.time()
                # print(loss)
                time_taken = (current - start)
                print("Epoch: {}, Epoch steps: {}, Global steps: {}, Time taken: {} seconds, Loss: {}".format(epoch_count, epoch_steps, global_training_steps, time_taken, train_loss))
            #perform evaluation every set amount of training steps
            if (global_training_steps % 5000 == 0):
                eval_loss, perplexity = evaluate(model, optimizer)
                eval_loss_list.append(eval_loss)
                print("Evaluation at Global step {} completed, Loss: {}, Perplexity {}".format(global_training_steps, eval_loss, perplexity))                
                # save the checkpoint model if the performance has improved
                # if (len(eval_loss_list) > 1):
                #     last_index = len(eval_loss_list) -1
                #     if (eval_loss_list[-1] < eval_loss_list[-2] and (epoch_count) > 2):
                output_directory = os.path.join(output_dir, "checkpoint-{}".format(global_training_steps))
                model.save_pretrained(output_directory)
                print("Checkpoint {} saved to directory {}".format(global_training_steps, output_directory))
        #at the end of each epoch, check if there has been improvement over the last 2 epochs to determine whether further training is needed
        ## could be done better         
        epoch_loss_list.append(eval_loss_list[-1])
        # if (len(epoch_loss_list) > 2):
        #     if (not(epoch_loss_list[-1] > epoch_loss_list[-2] and epoch_loss_list[-1] > epoch_loss_list[-3])):
        #         print("No improvement in model over the last 2 epochs, concluding training")
        #         return eval_loss_list
        epoch_count +=1
    print("Finished training!")


def evaluate(model):

    validation_data = []
    with open(validation_data_directory, 'rb') as f:
        validation_data = pickle.load(f)
        print("Evaluation dataset loaded")
    validation_dataset = dataset.ESCDataset(validation_data)
    dataloader = DataLoader(validation_dataset, shuffle=True, batch_size = batch_size)

    eval_loss = 0.0
    eval_steps = 0.0
    model.eval()
    print("Evaluation starting")
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs[0]
        eval_loss += loss.mean().item()
        # print(eval_loss)
        eval_steps += 1
    print("Finished Evaluation")
    eval_loss = eval_loss/eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    return eval_loss, perplexity

def inference(text_string, model, tokenizer):  
    input_ids = tokenizer(text_string, return_tensors='pt').to(device)
    # print(input_ids)
    output = model.generate(**input_ids, max_new_tokens=100,
            top_p=0.9,
            temperature=0.6,
            do_sample=True,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id)
    # print(output)

    pred = tokenizer.decode(output[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
    print("Model's Response: {}".format(pred))
    return tokenizer.decode(output[0])

def test(model, tokenizer):
    test_data = []
    with open(test_data_directory, 'rb') as f:
        test_data = pickle.load(f)
        print("test dataset loaded")
    test_dataset = dataset.ESCDataset(test_data)
    dataloader = DataLoader(test_dataset, shuffle=False, batch_size = batch_size)
    test_output = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        output = model.generate(**input_ids, max_new_tokens=100)
        pred = tokenizer.decode(output[0][input_ids.input_ids.shape[1]:])
        test_output.append(pred)
    print(tokenizer.decode(test_output[0]))
    return test_data, test_output


def metrics(results, labels):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bleu_results = bleu.compute(predictions=results, references=labels, max_order=2)
    rouge_results = rouge.compute(predictions=results, references=labels, rouge_types=['rougeL'])
    return bleu_results, rouge_results

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if(checkpoint != ""):
        model_path = checkpoint
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    sepcial_tokens_dict = {'additional_special_tokens': ['<skr>', '<sup>']}
    tokenizer.add_special_tokens(sepcial_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Training
    if (mode  == 0):  
        print("Training mode started")
        print("Model Loaded")
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
        results = train(model, optimizer)
        del model
    # Inference
    if (mode == 1): 
        print("Inference mode started")
        print("Please select the role you would like to play between Supporter and Seeker: ")
        role = input().lower()
        return_string = ""
        while(True):
            print("Please enter text to converse with the model: (enter \"exit\" to end conversation)" )
            text_string = input()
            if(text_string == "exit"):
                break
            if (role == "supporter"):
                text_string = return_string + "<sup>" + text_string + "<|endoftext|>"
            else:
                text_string = return_string + "<skr>" + text_string + "<|endoftext|>"
            return_string = inference(text_string, model, tokenizer)
            print("The conversation thus far, including special tokens: \n {}".format(return_string))
        print("Finished conversation with gpt")
    if (mode == 2):
        print("Test mode started")
        test_input, test_output = test(mode, tokenizer)
        
        
        
    gc.collect()
    torch.cuda.empty_cache()

