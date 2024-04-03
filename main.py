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

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="fine tune")
parser.add_argument("--train", type=bool, default=False)
parser.add_argument("--eval", type=bool, default=False)
parser.add_argument("--infer", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_path", type=str, default="/data/sam/Chris/gpt-2/gpt2")
parser.add_argument("--num_epochs", type=int, default=4)
parser.add_argument("--device", type=int, default=-1)
parser.add_argument("--output_directory", type=str, default= "/data/sam/Chris/gpt-2")
parser.add_argument("--checkpoint_load", type=bool, default=False)
args = parser.parse_args()

mode =args.mode 
batch_size = args.batch_size
train = args.train
eval = args.eval
infer = args.infer
num_epochs = args.num_epochs
model_path = args.model_path #"/data/sam/Chris/gpt-2/gpt2"
output_dir = args.output_directory
checkpoint_load = args.checkpoint_load
data_directory = "./ESC_tokenized.pkl"
train_data_directory = "./ESC_train.pkl"
test_data_directory = "./ESC_test.pkl"
device = torch.device(f"cuda:{args.device}" if args.device >=0 else "cpu")



# print(loaded_data[0].tokens)
# print(train_data)
# print(test_data['input_ids'])



# print(len(train_dataset))
# print(len(test_dataset))


def train(model, optimizer):
    global_training_steps = 0
    if(checkpoint_load == True):
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
                if (len(eval_loss_list) > 1):
                    last_index = len(eval_loss_list) -1
                    if (eval_loss_list[-1] < eval_loss_list[-2] and (epoch_count) > 2):
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


def evaluate(model, optimizer):

    test_data = []
    with open(test_data_directory, 'rb') as f:
        test_data = pickle.load(f)
        print("Evaluation dataset loaded")
    test_dataset = dataset.ESCDataset(test_data)
    dataloader = DataLoader(test_dataset, shuffle=True, batch_size = batch_size)

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
    output = model.generate(**input_ids, max_new_tokens=500,
            top_p=0.9,
            temperature=0.6,
            do_sample=True,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id)
    # print(output)
    pred = tokenizer.decode(output[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
    print("Gpt-2's Response: {}".format(pred))
    print("Finished conversation with gpt")


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("/data/sam/Chris/gpt-2/gpt2")
    sepcial_tokens_dict = {'additional_special_tokens': ['<skr>', '<sup>']}
    tokenizer.add_special_tokens(sepcial_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))


    if (train is True): 
        print("Model Loaded")
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
        results = train(model, optimizer)
        del model
    if (infer is True): 
        print("Please enter text to converse with the model")
        text_string = input()
        text_string = "<skr>" + text_string + "<|endoftext|>"
        inference(text_string, model, tokenizer)
    gc.collect()
    torch.cuda.empty_cache()

