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

def split_into_sents(tokens):
    sents = []
    temp=[]
    for t in tokens:
        if t.endswith("."):
            temp.append(t)
            sents.append(temp)
            temp=[]
        else:
            temp.append(t)

    return sents
