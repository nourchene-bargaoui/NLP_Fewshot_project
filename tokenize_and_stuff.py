import os
from transformers import BertTokenizer

def get_tokens_and_labels(filename):
    tokens = []
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        print(filename)
        text = f.read()
        lines = text.split("\n")
        for line in lines:
            if len(line.split("\t"))==2:
                tokens.append(line.split("\t")[0])
                # postag.append(line.split("\t")[1])  Do we want POS Tags for something?
                labels.append(line.split("\t")[1])
            #else:
                #print(line)

    return tokens, labels

def split_into_sents(tokens, labels): #split list of toks ad labels into list of sentence lists
    sents = []
    temp=[] #used to accumulate sentence

    sent_labels = []
    temp2 = []# used to accumulate labels
    for i in range(len(tokens)):
        t = tokens[i]#get token
        l = labels[i]#get label
        if t.endswith("."):
            temp.append(t)
            sents.append(temp)
            temp=[]
            temp2.append(l)
            sent_labels.append(temp2)
            temp2=[]
        else:
            temp.append(t)
            temp2.append(l)
    if len(temp) != 0:
        sents.append(temp)
        sent_labels.append(temp2)

    return sents, sent_labels

def get_unique_labels(): #FOR TESTING ONLY
    unique = []
    for filepath in os.listdir("preprocessed_data/train/"):
        words, labels = get_tokens_and_labels("preprocessed_data/train/"+filepath)
        for label in labels:
            if label not in unique:
                unique.append(label)
    print(unique)
    print(len(unique))

def get_test_model_inputs(filename_to_t_and_l, max_len):
    output = {}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_dict = {"O":0, #it's like a rainbow, mapping labels to label ids
                  "B-EXAMPLE_LABEL":1,
                    "I-EXAMPLE_LABEL":2,
                      "B-REACTION_PRODUCT":3,
                        "I-REACTION_PRODUCT":4,
                          "B-STARTING_MATERIAL":5,
                            "I-STARTING_MATERIAL":6,
                              "B-REAGENT_CATALYST":7,
                                "I-REAGENT_CATALYST":8,
                                  "B-SOLVENT":9,
                                    "I-SOLVENT":10,
                                     "B-OTHER_COMPOUND":11, 
                                       "I-OTHER_COMPOUND":12,
                                         "B-TIME":13,
                                          "I-TIME":14,
                                           "B-TEMPERATURE":15,
                                            "I-TEMPERATURE":16, 
                                             "B-YIELD_OTHER":17,
                                               "I-YIELD_OTHER":18, 
                                                "B-YIELD_PERCENT":19,
                                                "I-YIELD_PERCENT":20,
                                                "B-REACTION_STEP":21,
                                                "I-REACTION_STEP":22,
                                                "B-WORKUP":23,
                                                "I-WORKUP":24,
                                                 "PAD":-100 }
    for filename in filename_to_t_and_l:
        attention_list = []
        ids_list = []
        labels_list = []
        tokens = filename_to_t_and_l[filename][0]
        labels = filename_to_t_and_l[filename][1]
        sents, sent_labels = split_into_sents(tokens, labels)
        for i in range(len(sents)):
            sent = sents[i]
            label_list = sent_labels[i]#get corresponding labels
            sent_attention = [1]
            sent_ids = [101]
            sent_label_ids = [0]
            counter = 1 #how many tokens are in the lists? are we over max_len? are we under then we need to pad
            for j in range(len(sent)): #for each word
                word = sent[j]
                l = label_list[j]#and label
                first=True #if it's the first token, it should be B-something and if it's not it should be I-something
                input_ids = tokenizer(word, return_tensors="pt", padding=True)['input_ids'].tolist()[0] #get those ids!!
                for id in input_ids:#for each id
                    if id!=101 and id != 102 and counter!=max_len-1:#we don't need 101 and 102 i'm manually adding them
                        if first:#add B-, label as is
                            first=False
                            sent_label_ids.append(label_dict[l])
                        else: # not B-
                            if label_dict[l] in [1,3,5,7,9,11,13,15,17,19,21,23]:
                                sent_label_ids.append(label_dict[l]+1)
                            else: #if it's already I- just add the label and move on
                                sent_label_ids.append(label_dict[l])
                        sent_ids.append(id)
                        sent_attention.append(1)
                        counter+=1#n3xt token
                    if counter==max_len-1:#at the end
                        sent_ids.append(102)#add sep token
                        sent_attention.append(1)
                        counter+=1
                        sent_label_ids.append(0)
                    if counter==max_len: #exit loop
                        break
            if counter<max_len-1:#if we haven't met the max_len, first add sep token id
                sent_ids.append(102)
                sent_attention.append(1)
                counter+=1
                sent_label_ids.append(0)
            for i in range(max_len-len(sent_ids)):#now we pad until max_len
                sent_ids.append(0)
                sent_attention.append(0)
                sent_label_ids.append(-100)
            attention_list.append(sent_attention)
            ids_list.append(sent_ids)
            labels_list.append(sent_label_ids)
        output[filename] = [ids_list, attention_list,labels_list]
    return output


def get_model_inputs(filename_to_t_and_l, max_len):
    total_attention_list=[]
    total_ids_list=[]
    total_labels_list = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")#laod tokenizer
    label_dict = {"O":0, #it's like a rainbow, mapping labels to label ids
                  "B-EXAMPLE_LABEL":1,
                    "I-EXAMPLE_LABEL":2,
                      "B-REACTION_PRODUCT":3,
                        "I-REACTION_PRODUCT":4,
                          "B-STARTING_MATERIAL":5,
                            "I-STARTING_MATERIAL":6,
                              "B-REAGENT_CATALYST":7,
                                "I-REAGENT_CATALYST":8,
                                  "B-SOLVENT":9,
                                    "I-SOLVENT":10,
                                     "B-OTHER_COMPOUND":11, 
                                       "I-OTHER_COMPOUND":12,
                                         "B-TIME":13,
                                          "I-TIME":14,
                                           "B-TEMPERATURE":15,
                                            "I-TEMPERATURE":16, 
                                             "B-YIELD_OTHER":17,
                                               "I-YIELD_OTHER":18, 
                                                "B-YIELD_PERCENT":19,
                                                "I-YIELD_PERCENT":20,
                                                "B-REACTION_STEP":21,
                                                "I-REACTION_STEP":22,
                                                "B-WORKUP":23,
                                                "I-WORKUP":24,
                                                 "PAD":-100 }
    for key in filename_to_t_and_l:#for each file
        file_ids = []#these are temp lists that we add as the value for the file in the dict above when we're done with the file
        file_attention_mask = []
        file_labels = []
        tokens =  filename_to_t_and_l[key][0]#get tokens
        labels = filename_to_t_and_l[key][1]#get labels
        sents, sent_labels = split_into_sents(tokens, labels)#group into sentences which look like [["75", "F", "sucks"],["another", "sentence", "here"]] and the corresponding labels
        for i in range(len(sents)):#go through each sentence
            sent = sents[i]
            label_list = sent_labels[i]#get corresponding labels
            ids = [101] #cls id, we'll use this array to store all the ids for this sent, we get the ids from the tokenizer
            attention_mask=[1]# attention mask because we're using padding
            label_ids = [0]# labels go here, just the number
            counter = 1 #how many tokens are in the lists? are we over max_len? are we under then we need to pad
            for j in range(len(sent)): #for each word
                word = sent[j]
                l = label_list[j]#and label
                first=True #if it's the first token, it should be B-something and if it's not it should be I-something
                input_ids = tokenizer(word, return_tensors="pt", padding=True)['input_ids'].tolist()[0] #get those ids!!
                for id in input_ids:#for each id
                    if id!=101 and id != 102 and counter!=max_len-1:#we don't need 101 and 102 i'm manually adding them
                        if first:#add B-, label as is
                            first=False
                            label_ids.append(label_dict[l])
                        else: # not B-
                            if label_dict[l] in [1,3,5,7,9,11,13,15,17,19,21,23]:
                                label_ids.append(label_dict[l]+1)
                            else: #if it's already I- just add the label and move on
                                label_ids.append(label_dict[l])
                        ids.append(id)
                        attention_mask.append(1)
                        counter+=1#n3xt token
                    if counter==max_len-1:#at the end
                        ids.append(102)#add sep token
                        attention_mask.append(1)
                        counter+=1
                        label_ids.append(0)
                    if counter==max_len: #exit loop
                        break
            if counter<max_len-1:#if we haven't met the max_len, first add sep token id
                ids.append(102)
                attention_mask.append(1)
                counter+=1
                label_ids.append(0)
            for i in range(max_len-len(ids)):#now we pad until max_len
                ids.append(0)
                attention_mask.append(0)
                label_ids.append(-100)
            total_ids_list.append(ids) #add ids to file list, same with attention mask and labels
            total_attention_list.append(attention_mask)
            total_labels_list.append(label_ids)
    return total_ids_list, total_attention_list, total_labels_list

def calculate_class_weights(label_dict, total_labels_list):
    weights = []
    for key in label_dict:
        total_k = 0
        total_samples= 0
        classk=label_dict[key]
        for sent in total_labels_list:
            for wordid in sent:
                if wordid==classk:
                    total_k+=1
                total_samples+=1
        weights.append(total_samples/(len(label_dict)*total_k))
        
    return  weights

