from tokenize_and_stuff import get_tokens_and_labels, split_into_sents, get_unique_labels
from transformers import BertTokenizer
from model import Model
import torch
import os
def main():
    model=Model()
    max_len=512
    batch_size=50
    filename_to_t_and_l = {}
    train_path = "preprocessed_data/train/"
    for filename in  os.listdir(train_path):  
        tokens, labels = get_tokens_and_labels(train_path+filename)
        filename_to_t_and_l[filename] = [tokens,labels]
    # print(filename_to_t_and_l)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_dict = {"O":0, #it's like a rainbow
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
    label_arr = ["O",
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
    
    filename_to_ids_attention_labels = {}
    for key in filename_to_t_and_l:
        file_ids = []
        file_attention_mask = []
        file_labels = []
        tokens =  filename_to_t_and_l[key][0]
        labels = filename_to_t_and_l[key][1]
        sents, sent_labels = split_into_sents(tokens, labels)
        for i in range(len(sents)):
            sent = sents[i]
            label_list = sent_labels[i]
            ids = [101]
            attention_mask=[1]
            label_ids = [0]
            counter = 1
            for j in range(len(sent)):
                word = sent[j]
                l = label_list[j]
                first=True
                input_ids = tokenizer(word, return_tensors="pt", padding=True)['input_ids'].tolist()[0]
                for id in input_ids:
                    if id!=101 and id != 102 and counter!=max_len-1:
                        if first:
                            first=False
                            label_ids.append(label_dict[l])
                        else:
                            if label_dict[l] in [1,3,5,7,9,11,13,15,17,19,21,23]:
                                label_ids.append(label_dict[l]+1)
                            else:
                                label_ids.append(label_dict[l])
                        ids.append(id)
                        attention_mask.append(1)
                        counter+=1
                    if counter==max_len-1:
                        ids.append(102)
                        attention_mask.append(1)
                        counter+=1
                        label_ids.append(0)
                    if counter==max_len:
                        break
            if counter<max_len-1:
                ids.append(102)
                attention_mask.append(1)
                counter+=1
                label_ids.append(0)
            for i in range(max_len-len(ids)):
                ids.append(0)
                attention_mask.append(0)
                label_ids.append(25)
            file_ids.append(ids)
            file_attention_mask.append(attention_mask)
            file_labels.append(label_ids)
        filename_to_ids_attention_labels[key] = [file_ids,file_attention_mask, file_labels]
    for key in filename_to_ids_attention_labels:
        ids = filename_to_ids_attention_labels[key][0]
        attention = filename_to_ids_attention_labels[key][1]
        batch_ids = [ids[j:j+batch_size] for j in range(0,len(ids), batch_size)]
        batch_attention = [attention[j:j+batch_size] for j in range(0,len(attention), batch_size)]
        batch_ids_tensor = torch.LongTensor(batch_ids)

        batch_attention_tensor = torch.LongTensor(batch_attention)
        outputs = model(batch_ids_tensor[0], batch_attention_tensor[0])
                    
if __name__=='__main__':
    main()