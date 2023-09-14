import argparse
import nltk
import os
import nltk
from nltk.tokenize import TreebankWordTokenizer as twt
from nltk import sent_tokenize
nltk.download("averaged_perceptron_tagger")
def parse_entities(filepath):
    entities = []
    with open(filepath, "r", encoding="utf-8") as entity_file:
        lines = entity_file.read().split("\n")
        for l in lines:
            if l.startswith("T"):
                temp = []
                temp.append(l.split("\t")[2])
                temp.append(l.split("\t")[1].split(" ")[0])
                temp.append((l.split("\t")[1].split(" ")[1],l.split("\t")[1].split(" ")[len(l.split("\t")[1].split(" "))-1]))
                entities.append(temp)
    return entities

parser = argparse.ArgumentParser(description="convert BRAT annotations to CONLL (BILOU)")
parser.add_argument("input_dir", help="directory with input files and annotations files. should be in form of *.txt and *.ann, and every .txt file should have a .ann file with the same name, e.g. 101.txt and 101.ann")
parser.add_argument("output_dir", help="directory to write output files to (.ann files in CONLL - BILOU format)")
args = parser.parse_args()

print("Checking input directory: "+args.input_dir)
for filename in os.listdir(args.input_dir):
    if filename.endswith(".txt") and not os.path.exists(args.input_dir+filename.split(".")[0]+".ann"):
        print("No associated .ann file for file: "+filename)
        exit()
print("Directory passed check.")
for filename in os.listdir(args.input_dir):
    if filename.endswith(".txt"):
        entities = parse_entities(args.input_dir+filename.split(".")[0]+".ann") #get the entities from brat file
        with open(args.input_dir+filename, "r", encoding="utf-8") as f:
            raw_text = f.read()
            spans = list(twt().span_tokenize(raw_text))
            print(spans)
            tokenized_text = []
            for s in spans:
                tokenized_text.append(raw_text[s[0]:s[1]])
            tagged_text = nltk.pos_tag(tokenized_text)
            with open(args.output_dir+filename.split(".")[0]+".ann","w",encoding="utf-8") as out:
                bflag = True
                for i in range(len(tagged_text)):
                    out.write(tagged_text[i][0]+"\t"+tagged_text[i][1]+"\t")
                    flag = True
                    for e in entities:
                        if int(e[2][0]) <= spans[i][0] and int(e[2][1]) >=spans[i][1]: #if its within the spans
                            print(spans[i][1])
                            print(e[2][1])
                            if spans[i][1]==int(e[2][1]):
                                if bflag:
                                    out.write("B-"+e[1]+"\n")
                                else:
                                    out.write("I-"+e[1]+"\n")
                                bflag=True
                                flag=False
                            elif bflag:
                                out.write("B-"+e[1]+"\n")
                                flag=False
                                bflag=False
                                break
                            else:
                                out.write("I-"+e[1]+"\n")
                                flag=False
                                bflag = False
                                break
                    if flag:
                        out.write("O"+"\n")
                        bflag=True
