from tokenize_and_stuff import get_tokens_and_labels, split_into_sents
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
    label_list = ["O", #it's like a rainbow
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
                                                "B-TIELD_PERCENT",
                                                "I-YIELD_PERCENT" ]
    
    filename_to_ids_attention = {}
    for key in filename_to_t_and_l:
        file_ids = []
        file_attention_mask = []
        tokens =  filename_to_t_and_l[key][0]
        labels = filename_to_t_and_l[key][1]
        sents, sent_labels = split_into_sents(tokens, labels)
        for sent in sents:
            ids = [101]
            attention_mask=[1]
            counter = 1
            for word in sent:
                input_ids = tokenizer(word, return_tensors="pt", padding=True)['input_ids'].tolist()[0]
                for id in input_ids:
                    if id!=101 and id != 102 and counter!=max_len-1:
                        ids.append(id)
                        attention_mask.append(1)
                        counter+=1
                    if counter==max_len-1:
                        ids.append(102)
                        attention_mask.append(1)
                        counter+=1
                    if counter==max_len:
                        break
            if counter<max_len-1:
                ids.append(102)
                attention_mask.append(1)
                counter+=1
            for i in range(max_len-len(ids)):
                ids.append(0)
                attention_mask.append(0)
            file_ids.append(ids)
            file_attention_mask.append(attention_mask)
        filename_to_ids_attention[key] = [file_ids,file_attention_mask]
    for key in filename_to_ids_attention:
        ids = filename_to_ids_attention[key][0]
        attention = filename_to_ids_attention[key][1]
        batch_ids = [ids[j:j+batch_size] for j in range(0,len(ids), batch_size)]
        batch_attention = [attention[j:j+batch_size] for j in range(0,len(attention), batch_size)]
        batch_ids_tensor = torch.LongTensor(batch_ids)
        batch_attention_tensor = torch.LongTensor(batch_attention)
        print(batch_attention_tensor[0].size())
        outputs = model(batch_ids_tensor[0], batch_attention_tensor[0])
                    
if __name__=='__main__':
    main()