import matplotlib.pyplot as plt

# Data dictionaries for each data type
data_dev = {
    'REACTION_STEP': 419, 'EXAMPLE_LABEL': 96, 'TIME': 117, 'SOLVENT': 120,
    'WORKUP': 331, 'OTHER_COMPOUND': 524, 'STARTING_MATERIAL': 180,
    'REAGENT_CATALYST': 150, 'TEMPERATURE': 163, 'REACTION_PRODUCT': 220,
    'YIELD_OTHER': 122, 'YIELD_PERCENT': 106
}

data_train = {
    'REACTION_STEP': 4233, 'EXAMPLE_LABEL': 982, 'TIME': 1176, 'SOLVENT': 1260,
    'WORKUP': 3384, 'OTHER_COMPOUND': 5164, 'STARTING_MATERIAL': 1934,
    'REAGENT_CATALYST': 1431, 'TEMPERATURE': 1678, 'REACTION_PRODUCT': 2272,
    'YIELD_OTHER': 1183, 'YIELD_PERCENT': 1061
}

data_test = {
    'REACTION_STEP': 6209, 'EXAMPLE_LABEL': 1452, 'TIME': 1763, 'SOLVENT': 1818,
    'WORKUP': 5026, 'OTHER_COMPOUND': 7650, 'STARTING_MATERIAL': 2877,
    'REAGENT_CATALYST': 2074, 'TEMPERATURE': 2473, 'REACTION_PRODUCT': 3411,
    'YIELD_OTHER': 1761, 'YIELD_PERCENT': 1571
}

# Create bar charts for each data type
def create_bar_chart(data, data_type):
    labels = data.keys()
    values = data.values()

    plt.figure(figsize=(10, 10))
    plt.bar(labels, values, linewidth=2, edgecolor='k')

    for i, v in enumerate(values):
        plt.text(i, v + 10, str(v), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title(f'{data_type} Data Type')
    plt.xticks(rotation=45)
    plt.tight_layout()


    plt.show()

# create_bar_chart(data_dev, 'dev')
create_bar_chart(data_train, 'train')
# create_bar_chart(data_test, 'test')
