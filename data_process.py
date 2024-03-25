import json 
import os
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
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
processed_data = []
#print(dataset)
for dialog in dataset["train"]:
    # print(dialog)
    dialog = dialog["dialog"]
    line = ""
    speaker = "seeker"
    for turn in dialog:
        if(speaker == "supporter" and turn["speaker"]):
            processed_data.append(line)
            # print(line) 
        speaker = turn["speaker"]
        if speaker == "seeker":
            line += seeker_token
        else: 
            line += supporter_token
        line += turn["content"].replace("\n", "")+"<eos>"
    processed_data.append(line)
    # break
    # for line in processed_data:
        # print(line)
# exit()
tokenizer = AutoTokenizer.from_pretrained("/data/sam/Chris/gpt-2/gpt2")
sepcial_tokens_dict = {'additional_special_tokens': ['<skr>', '<sup>']}
tokenizer.add_special_tokens(sepcial_tokens_dict)
tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.add_special_tokens({'eos_token': '<eos>'})
print("length of tokenizer is {}".format(len(tokenizer)))
# print(processed_data.items())
# exit()
train_data, test_data = train_test_split(processed_data,test_size=.2)

# print(tokenizer(processed_data)[1].tokens)
tokenized_train_data = tokenizer(train_data, padding=True, truncation=True)
tokenized_test_data = tokenizer(test_data, padding=True, truncation=True)
# print(len(tokenized_data[0]))
with open('ESC_train.pkl', 'wb') as f:
    pickle.dump(tokenized_train_data, f)
with open('ESC_test.pkl', 'wb') as f:
    pickle.dump(tokenized_test_data, f)
# with open('ESC_tokenized.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)
#     print(loaded_data[0].tokens)
# tokenized_data = dataset.map(lambda dataset: tokenizer(processed_data))
# tokenized_data.save_to_disk('.')
# f = open("./processed_data.txt", "w")
# for line in processed_data:
    # f.write(line + "\n")
