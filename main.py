from tokenize import get_tokens_and_labels 
from transformers import BertTokenizer
import os
def main():
    filename_to_t_and_l = {}
    train_path = "preprocessed_data/train/" 
    for filename in  os.listdir(train_path):  
        tokens, labels = get_tokens_and_labels(train_path+filename)
        filename_to_t_and_l[filename] = [tokens,labels]
    print(filename_to_t_and_l)
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

if __name__=='__main__':
    main()