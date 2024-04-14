import json 
import os
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_directory", type=str, default="/data/sam/Chris/gpt-2/gpt2")
parser.add_argument("--save_directory", type=str, default="./data/gpt2")
args = parser.parse_args()
MODEL_DIRECTORY = args.model_directory
SAVE_DIRECTORY = args.save_directory

data_directory = "/data/sam/Chris/ESC/ESConv.json"
# dataset = load_dataset("json", data_files=data_directory)
dataset = load_dataset("json", data_files= data_directory)
# print(len(dataset))
# print(len(dataset[0]["dialog"]))
seeker_token = "<skr>"
supporter_token = "<sup>"
# for dialog in dataset:
#     dialog = dialog["dialog"]
#     for turn in dialog:
#         print(turn["speaker"] + ": " + turn["content"])
#     break
count= 0
train_split = []
test_split = []
for i in range(0, 1040):
    train_split.append(dataset['train'][i])
for i in range(1040, 1300):
    test_split.append(dataset['train'][i])
# train_split, test_split = train_test_split(dataset["train"],test_size=.2)
# print(train_split)
# print(test_split)
#print(dataset)
def process(dataset):
    processed_data = []
    for dialog in dataset:
        # print(dialog)
        dialog = dialog["dialog"]
        line = ""
        speaker = ""
        last_speaker = ""
        for turn in dialog:
            speaker = turn['speaker']
            if(not(speaker == last_speaker)):
                if (not(last_speaker == "")):
                    line +=tokenizer.eos_token
                    # if(speaker == "supporter"): ### if we want it to learn only 1 role
                    processed_data.append(line)
                if speaker == "seeker":
                    line += seeker_token
                else: 
                    line += supporter_token
            line += turn["content"].replace("\n", "").lower()
            last_speaker = speaker
        line += tokenizer.eos_token
        processed_data.append(line)
    return processed_data
    # break
    # for line in processed_data:
        # print(line)
# exit()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRECTORY)
train_data = process(train_split)
test_data = process(test_split)
sepcial_tokens_dict = {'additional_special_tokens': ['<skr>', '<sup>']}
tokenizer.add_special_tokens(sepcial_tokens_dict)
# print(tokenizer("<skr>Hello").tokens)
# print(tokenizer("<skr> Hello").tokens)
# exit()
print("length of tokenizer is {}".format(len(tokenizer)))
# print(processed_data.items())
# print(train_data)
# print(test_data)
# print(tokenizer(processed_data)[1].tokens)
tokenized_train_data = tokenizer(train_data, truncation=True)
tokenized_test_data = tokenizer(test_data, truncation=True)
# print(len(tokenized_data[0]))



with open(os.path.join(SAVE_DIRECTORY, 'ESC_train.pkl'), 'wb') as f:
    pickle.dump(tokenized_train_data, f)
with open(os.path.join(SAVE_DIRECTORY, 'ESC_test.pkl'), 'wb') as f:
    pickle.dump(tokenized_test_data, f)


# with open('ESC_tokenized.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)
#     print(loaded_data[0].tokens)
# tokenized_data = dataset.map(lambda dataset: tokenizer(processed_data))
# tokenized_data.save_to_disk('.')

with open(os.path.join(SAVE_DIRECTORY,"processed_train_data.txt"), "w") as f:
    for line in train_data:
        f.write(line + "\n")
with open(os.path.join(SAVE_DIRECTORY,"processed_test_data.txt"), "w") as f:
    for line in test_data:
        f.write(line + "\n")


