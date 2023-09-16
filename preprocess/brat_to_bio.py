import argparse
import nltk
import os
import nltk
from nltk.tokenize import TreebankWordTokenizer as twt
from nltk import sent_tokenize
nltk.download("averaged_perceptron_tagger")
def parse_entities(filepath):#get the entitites and spans and text
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
#cli stuff
parser = argparse.ArgumentParser(description="convert BRAT annotations to CONLL (BILOU)")
parser.add_argument("input_dir", help="directory with input files and annotations files. should be in form of *.txt and *.ann, and every .txt file should have a .ann file with the same name, e.g. 101.txt and 101.ann")
parser.add_argument("output_dir", help="directory to write output files to (.ann files in CONLL - BILOU format)")
args = parser.parse_args()
#check to make sure every txt file has an ann file
print("Checking input directory: "+args.input_dir)
for filename in os.listdir(args.input_dir):
    if filename.endswith(".txt") and not os.path.exists(args.input_dir+filename.split(".")[0]+".ann"):
        print("No associated .ann file for file: "+filename)
        exit()
print("Directory passed check.")
for filename in os.listdir(args.input_dir):#for each file
    if filename.endswith(".txt"):#if it's a txt file
        entities = parse_entities(args.input_dir+filename.split(".")[0]+".ann") #get the entities from brat file
        with open(args.input_dir+filename, "r", encoding="utf-8") as f:#read the txt file
            raw_text = f.read()
            spans = list(twt().span_tokenize(raw_text))#tokenize and get spans
            print(spans)
            tokenized_text = []#put tokens in here
            for s in spans:
                tokenized_text.append(raw_text[s[0]:s[1]])
            tagged_text = nltk.pos_tag(tokenized_text)#pos tag here
            with open(args.output_dir+filename.split(".")[0]+".ann","w",encoding="utf-8") as out:#open ann file
                bflag = True#tag as B, start of entity
                for i in range(len(tagged_text)):#go tthrough the tag text
                    out.write(tagged_text[i][0]+"\t"+tagged_text[i][1]+"\t") #output the text and the pos tag
                    flag = True #whether or not to output "O"
                    for e in entities: #search entities
                        if int(e[2][0]) <= spans[i][0] and int(e[2][1]) >=spans[i][1]: #if its within the spans
                            print(spans[i][1])#testing print
                            print(e[2][1])#testing print
                            if spans[i][1]==int(e[2][1]):#do the ending spans match
                                if bflag:#if ti's the first one
                                    out.write("B-"+e[1]+"\n")
                                else:#if it's not the first one
                                    out.write("I-"+e[1]+"\n")
                                bflag=True#we've ended the entity so the next should be b
                                flag=False#we outputted so no "O"
                            elif bflag:#if the ending spans dont match and it's the first one
                                out.write("B-"+e[1]+"\n")
                                flag=False
                                bflag=False
                                break
                            else: #otherwise it's an I
                                out.write("I-"+e[1]+"\n")
                                flag=False
                                bflag = False
                                break
                    if flag: #if we didn't write any label, label it as "O"
                        out.write("O"+"\n")
                        bflag=True
