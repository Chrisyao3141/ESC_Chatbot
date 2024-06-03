
def distinct(tokens):
    distinct_list = []
    distinct_count = 0
    total_count = 0
    for sequence in tokens:
        for token in sequence:
            if token in distinct_list:
                total_count+=1
            else:
                distinct_list.append(token)
                distinct_count+=1
                total_count+=1
    return distinct_count/total_count

token_list = [["test", "this", "is", "apple"], ["I", "like ", "to", "eat", "apple"]]
print(f"The Distinct score is: {distinct(token_list)}")