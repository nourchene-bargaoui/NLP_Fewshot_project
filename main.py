from tokenize_and_stuff import get_tokens_and_labels, split_into_sents
from transformers import BertTokenizer
import torch
import os
def main():
    max_len=512
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
    
    
    for key in filename_to_t_and_l:
        file_ids = []
        file_attention_mask = []
        tokens =  filename_to_t_and_l[key][0]
        sents = split_into_sents(tokens)
        print(sents)
        for sent in sents:
            ids = [101]
            attention_mask=[1]
            counter = 1
            for word in sent:
                input_ids = tokenizer(word, return_tensors="pt", padding=True)[input_ids][0]
                for id in input_ids:
                    
if __name__=='__main__':
    main()