
def EAD_distinct(tokens, v_size):
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
    print(distinct_count)
    print(total_count)
    return distinct_count/(v_size*(1-((v_size-1)/v_size)**total_count))

token_list = [["test", "this", "is", "apple"], ["I", "like ", "to", "eat", "apple"]]
print(f"The Distinct score is: {EAD_distinct(token_list, 800)}")