import os
def get_tokens_and_labels(filename):
    tokens = []
    labels = []
    with open(filename, "r") as f:
        text = f.read()
        lines = text.split("\n")
        for line in lines:
            if len(line.split("\t"))==3:
                tokens.append(line.split("\t")[0])
                # postag.append(line.split("\t")[1])  Do we want POS Tags for something?
                labels.append(line.split("\t")[2])
            else:
                print(line)
    return tokens, labels

def split_into_sents(tokens, labels):
    sents = []
    temp=[]

    sent_labels = []
    temp2 = []
    for i in range(len(tokens)):
        t = tokens[i]
        l = labels[i]
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

def get_unique_labels():
    unique = []
    for filepath in os.listdir("preprocessed_data/train/"):
        words, labels = get_tokens_and_labels("preprocessed_data/train/"+filepath)
        for label in labels:
            if label not in unique:
                unique.append(label)
    print(unique)
    print(len(unique))