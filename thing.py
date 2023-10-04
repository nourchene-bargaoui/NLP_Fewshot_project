import os
import matplotlib.pyplot as plt
path = "/home/ndil/Downloads/ChEMU2023_FOR_CLASS/dev/"
dict = {"REACTION_STEP":0,
        "EXAMPLE_LABEL":0,
        "TIME":0,
        "SOLVENT":0,
        "WORKUP":0,
        "OTHER_COMPOUND":0,
        "STARTING_MATERIAL":0,
        "REAGENT_CATALYST":0,
        "TEMPERATURE":0,
        "REACTION_PRODUCT":0,
        "YIELD_OTHER":0,
        "YIELD_PERCENT":0}
for filename in os.listdir(path):
    if filename.endswith(".ann"):
        with open(path+filename, "r") as f:
            lines = f.read().split("\n")
            for l in lines:
                if l!="" and l.startswith("T"):
                    tokens = l.split("\t")
                    entity= tokens[1].split(" ")[0]
                    if entity not in dict:
                        dict[entity] = 1
                    else:
                        dict[entity]+=1
print(dict)

plt.bar(list(dict.keys()), list(dict.values()), width=.4)
plt.xticks(rotation='vertical')
plt.xlabel("Entities")
plt.ylabel("Counts")
plt.title("Dev Data")
plt.subplots_adjust(bottom=0.4)
plt.savefig("dev.png")