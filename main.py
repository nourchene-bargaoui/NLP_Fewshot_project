from tokenize_and_stuff import get_tokens_and_labels, split_into_sents, get_unique_labels
from transformers import BertTokenizer
from model import Model
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
def main():
    model=Model() #our model from model.py
    max_len=512 #max sequence length, this is bert's max
    batch_size=1 #batch size for the model, 15 max
    filename_to_t_and_l = {} #mapping file names to tokens and labels
    train_path = "preprocessed_data/train/" #path to training, we should take this from CLI
    epochs = 1 #number of epochs, how many times do we iterate over the dataset?
    for filename in  os.listdir(train_path): #for each training file...
        tokens, labels = get_tokens_and_labels(train_path+filename) #getting the tokens and corresponding labels
        filename_to_t_and_l[filename] = [tokens,labels]#map to filename
    # print(filename_to_t_and_l)
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
                                                 "PAD":25 }
    label_arr = ["O", #mapping ids to labels
                 "B-EXAMPLE_LABEL",
                  "I-EXAMPLE_LABEL",
                      "B-REACTION_PRODUCT",
                        "I-REACTION_PRODUCT",
                          "B-STARTING_MATERIAL",
                            "I-STARTING_MATERIAL",
                              "B-REAGENT_CATALYST",
                                "I-REAGENT_CATALYST",
                                  "B-SOLVENT",
                                    "I-SOLVENT",
                                     "B-OTHER_COMPOUND", 
                                       "I-OTHER_COMPOUND",
                                         "B-TIME",
                                          "I-TIME",
                                           "B-TEMPERATURE",
                                            "I-TEMPERATURE", 
                                             "B-YIELD_OTHER",
                                               "I-YIELD_OTHER", 
                                                "B-YIELD_PERCENT",
                                                "I-YIELD_PERCENT",
                                                "B-REACTION_STEP",
                                                "I-REACTION_STEP",
                                                "B-WORKUP",
                                                "I-WORKUP",
                                                "PAD"]
    
    filename_to_ids_attention_labels = {}#map filenames to IDS, attention mask, subword_labels
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
                label_ids.append(25)
            file_ids.append(ids) #add ids to file list, same with attention mask and labels
            file_attention_mask.append(attention_mask)
            file_labels.append(label_ids)
        filename_to_ids_attention_labels[key] = [file_ids,file_attention_mask, file_labels] #map lists to file
    #here i realized that the files don't actually matter but i was too lazy to fix the code so i just iterate through them add add them to a total list
    total_attention_list=[]
    total_ids_list=[]
    total_labels_list = []
    for key in filename_to_ids_attention_labels:
        ids = filename_to_ids_attention_labels[key][0]
        for i in ids:
            total_ids_list.append(i)

        attention = filename_to_ids_attention_labels[key][1]
        for a in attention:
            total_attention_list.append(a)
        labels_list = filename_to_ids_attention_labels[key][2]
        for l in labels_list:
            total_labels_list.append(l)
    #convert to long tensors and add to dataset -> dataloader. shuffle and set batch_size
    train_set = TensorDataset(torch.LongTensor(total_ids_list), torch.LongTensor(total_attention_list), torch.LongTensor(total_labels_list))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for i in range(epochs): #do ___ epochs
        print(len(train_loader))#testing print statement
        for t, a, l in train_loader:# for each token ids, attention mask, and labels
            outputs = model(t, a)# feed into the model and get output!
                    
if __name__=='__main__':
    main()