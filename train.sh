nohup python -u main.py --mode 0 --model_path $gpt2 --num_epochs 10 --device 6 --data_directory ./data/gpt2  | tee train_log.txt

nohup python -u main.py --mode 2 --model_path $gpt2  --device 3 --data_directory ./data/gpt2 | tee test_log.txt