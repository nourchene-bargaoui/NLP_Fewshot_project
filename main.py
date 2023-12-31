"""
Charlie Dil, Masrik Dahir, Nourchen Bargaoui - 9/27/2023

Problem Statement: Given multiple texts, we want to be able to automatically identify key entities to use potentially in a larger NLP Task like LBD
This project is to compare our fewshot system with.

Example of input file:
Example	O
194	B-EXAMPLE_LABEL
3-Isobutyl-5-methyl-1-(oxetan-2-ylmethyl)-6-[(2-oxoimidazolidin-1-yl)methyl]thieno[2,3-d]pyrimidine-2,4(1H,3H)-dione	B-REACTION_PRODUCT
(racemate)	I-REACTION_PRODUCT
813	O
...

Example of output (classification report)
                    precision    recall  f1-score   support

                  O       1.00      0.99      0.99     66273
    B-EXAMPLE_LABEL       0.97      0.99      0.98       471
    I-EXAMPLE_LABEL       0.89      0.96      0.92       150
 B-REACTION_PRODUCT       0.92      0.96      0.94      1119
 I-REACTION_PRODUCT       0.98      0.98      0.98     28934
B-STARTING_MATERIAL       0.77      0.95      0.85       944
I-STARTING_MATERIAL       0.98      0.97      0.97     19214
 B-REAGENT_CATALYST       0.90      0.93      0.91       643
 I-REAGENT_CATALYST       0.90      0.92      0.91      2845
          B-SOLVENT       0.93      0.96      0.95       557
          I-SOLVENT       0.95      0.97      0.96      1607
   B-OTHER_COMPOUND       0.96      0.97      0.96      2487
   I-OTHER_COMPOUND       0.94      0.92      0.93      8663
             B-TIME       0.99      1.00      1.00       587
             I-TIME       0.99      1.00      1.00       674
      B-TEMPERATURE       0.98      1.00      0.99       795
      I-TEMPERATURE       0.97      0.99      0.98      1255
      B-YIELD_OTHER       0.98      1.00      0.99       579
      I-YIELD_OTHER       0.99      1.00      0.99      1809
    B-YIELD_PERCENT       1.00      1.00      1.00       506
    I-YIELD_PERCENT       1.00      1.00      1.00       793
... and the rest

Usage instructions: See README.md

Architecture/Algorithm:

We use regular BERT for contextualized representations, which we pass into a BiLSTM and then into a linear layer to get a sequence of labels
"""


#import statments
from tokenize_and_stuff import get_tokens_and_labels, split_into_sents, get_unique_labels, get_model_inputs, calculate_class_weights, get_test_model_inputs
from transformers import BertTokenizer
from model import NERModel
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import f1_score, accuracy_score, classification_report # ****
from torch import optim # ****
import matplotlib.pyplot as plt# ****
import argparse
import os


def main():
    
    #command line args
    parser = argparse.ArgumentParser(
                    prog='python main.py',
                    description='Baseline NER system for CHEMU dataset')
    parser.add_argument("--train_path", help="Path to training data, required if train is true")
    parser.add_argument("--dev_path", help="path to the dev data for validation. required if train is true")
    parser.add_argument("--test_path", help="path to the test data, required if test is true")
    parser.add_argument("--train", action="store_true", help="train flag, use if training")
    parser.add_argument("--test", action="store_true", help="test flag, use if you want to test")
    parser.add_argument("--model", help="specify path to model, required if test is true and train is false.")
    parser.add_argument("--epochs", help="specify the number of epochs, required if train is true")
    args = parser.parse_args()

    #print out command line args
    print("Train path: ", args.train_path)
    print("Test path: ", args.test_path)
    print("Dev path: ", args.dev_path)
    print("Train flag: ", args.train)
    print("Test flag: ", args.test)
    print("model path: ", args.model)
    train = args.train
    test = args.test
    train_path=""
    dev_path = ""
    test_path=""
    model_path=""
    epochs = 0
    device="cuda:0" #for gpu
    if not train and not test: #error messages for command line args
       print("ERROR: You need to either pick train or test")
       exit()
    if train:
      if args.train_path is None:
         print("ERROR: need train path to be specified to train")
         exit()
      if args.dev_path is None:
         print("ERROR: need dev path to be specified to train")
         exit()
      if args.epochs is None:
         print("ERROR: need epochs to be specified to train")
         exit()
      train_path = args.train_path
      dev_path=args.dev_path
      epochs = int(args.epochs)
    if test:
      if not train:
        if args.model is None:
          print("ERROR: model path cannot be none if train is false and test is true")
          exit()
        model_path = args.model
          
      if args.test_path is None:
        print("ERROR: Test path cannot be none for test==True")
        exit()
      test_path = args.test_path
     #path to training, we should take this from CLI
    num_classes = 25  # Update this with the actual number of classes
    # model=Model() #our model from model.py
    model = NERModel(num_classes)  # Create the NER model with BiLSTM from model.py
    model.to(device)
    max_len=512 #max sequence length, this is bert's max
    batch_size=10 #batch size for the model, 15 max
    filename_to_t_and_l = {} #mapping file names to tokens and labels
    filename_to_t_and_l_dev = {}

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
                                                "I-WORKUP":24}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    train_loss_history = []  # To store training loss ****
    val_loss_history = []    # To store validation loss********
    val_f1_history = []      # To store validation F1 score********
    val_accuracy_history = []  # To store validation accuracy**********

    if train:
      for filename in  os.listdir(train_path): #for each training file...
          if filename.endswith(".connl"):
            tokens, labels = get_tokens_and_labels(train_path+filename) #getting the tokens and corresponding labels
            filename_to_t_and_l[filename] = [tokens,labels]#map to filename
      
      for filename in os.listdir(dev_path):
          if filename.endswith(".connl"):
            tokens, labels = get_tokens_and_labels(dev_path+filename)
            filename_to_t_and_l_dev[filename] = [tokens, labels]
      # print(filename_to_t_and_l)
      tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")#laod tokenizer

      total_ids_list, total_attention_list, total_labels_list = get_model_inputs(filename_to_t_and_l, max_len)
      total_ids_list_dev, total_attention_list_dev, total_labels_list_dev = get_model_inputs(filename_to_t_and_l_dev, max_len)
      #convert to long tensors and add to dataset -> dataloader. shuffle and set batch_size

      train_set = TensorDataset(torch.LongTensor(total_ids_list), torch.LongTensor(total_attention_list), torch.LongTensor(total_labels_list))
      train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

      val_set = TensorDataset(torch.LongTensor(total_ids_list_dev), torch.LongTensor(total_attention_list_dev), torch.LongTensor(total_labels_list_dev))
      val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
      
      weight = calculate_class_weights(label_dict, total_labels_list)
      loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weight)) # our loss function !! now with weights
      # ******************************************************************************************
      loss_function = loss_function.to(device)
      print("Training beginning now")
      for epoch in range(epochs): #for e epochs
          model.train()
          total_loss = 0.0

          for input_ids, attention_mask, labels in train_loader: #for each batch
              input_ids = input_ids.to(device) # send to gpu
              attention_mask = attention_mask.to(device)
              labels = labels.to(device)
              optimizer.zero_grad()
              logits = model(input_ids, attention_mask)
              loss = loss_function(logits.view(-1, num_classes), labels.view(-1))  # Flatten logits and labels
              loss.backward()
              optimizer.step()
              total_loss += loss.item()
              #break debugging only

          avg_loss = total_loss / len(train_loader)
          train_loss_history.append(avg_loss) # for the plot !

          # Evaluation on validation set
          model.eval()
          # ... Perform evaluation and calculate F1-score or other metrics ...
          val_loss = 0.0
          all_predictions = []
          all_labels = []
          with torch.no_grad():
              for input_ids, attention_mask, labels in val_loader:
                  input_ids = input_ids.to(device)
                  attention_mask = attention_mask.to(device)
                  labels = labels.to(device)
                  logits = model(input_ids, attention_mask)
                  loss = loss_function(logits.view(-1, num_classes), labels.view(-1))
                  val_loss += loss.item()

                  predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                  labels = labels.cpu().numpy()
                  all_predictions.extend(predictions[labels != -100])
                  all_labels.extend(labels[labels != -100])
                  #break debugging only
          

          avg_val_loss = val_loss / len(val_loader)
          val_loss_history.append(avg_val_loss)

          val_f1 = f1_score(all_labels, all_predictions, average='macro', labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
          val_f1_history.append(val_f1)
          val_accuracy = accuracy_score(all_labels, all_predictions)
          val_accuracy_history.append(val_accuracy)
          
          print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation F1: {val_f1:.4f}, Validation Accuracy: {val_accuracy:.4f}")

      # Plot training history
          torch.save(model.state_dict(), "model_dumps/epoch"+str(epoch)+".pth")
      plt.figure(figsize=(12, 4))
      plt.subplot(1, 3, 1)
      plt.plot(train_loss_history, label='Train Loss')
      plt.plot(val_loss_history, label='Validation Loss')
      plt.legend()
      plt.xlabel('Epoch')
      plt.ylabel('Loss')

      plt.subplot(1, 3, 2)
      plt.plot(val_f1_history, label='Validation F1 Score')
      plt.legend()
      plt.xlabel('Epoch')
      plt.ylabel('F1 Score')

      plt.subplot(1, 3, 3)
      plt.plot(val_accuracy_history, label='Validation Accuracy')
      plt.legend()
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')

      plt.tight_layout()
      plt.savefig("thing.png")
      #plt.show()
      plt.close()
    if test: #eval ccode
        if not train: # if nto train we have to load
          model = NERModel(num_classes)
          model.load_state_dict(torch.load(model_path))
          model = model.to(device)
        model.eval()
        filename_to_t_and_l_test = {}
        for filename in os.listdir(test_path):
          if filename.endswith(".connl"):
            tokens, labels = get_tokens_and_labels(test_path+filename)
            filename_to_t_and_l_test[filename] = [tokens, labels]
        test_dict = get_test_model_inputs(filename_to_t_and_l_test, max_len)
        all_predictions = []
        all_labels = []
        all_label_ids = []
        all_attention_masks = []
        all_ids = []

        for filename in test_dict:
            all_ids.extend(test_dict[filename][0])
            all_attention_masks.extend(test_dict[filename][1])
            all_label_ids.extend(test_dict[filename][2])
        test_set = TensorDataset(torch.LongTensor(all_ids), torch.LongTensor(all_attention_masks), torch.LongTensor(all_label_ids))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        i=1
        for input_ids, attention_mask, labels in test_loader: #for batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            print(str(i)+"/"+str(len(test_loader)))
            i+=1
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_predictions.extend(predictions[labels!=-100])
            all_labels.extend(labels[labels!=-100])
        # print(all_predictions)
        # print(all_labels)
        with open("test_report.txt", "w") as out: #write out scores
            out.write(classification_report(all_labels, all_predictions, target_names = list(label_dict.keys()), labels=list(label_dict.values())))
        
if __name__=='__main__':
    main()
