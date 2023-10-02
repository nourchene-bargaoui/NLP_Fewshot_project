import pandas as pd

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

# Create dataframes for each data type
df_dev = pd.DataFrame.from_dict(data_dev, orient='index', columns=['Count'])
df_train = pd.DataFrame.from_dict(data_train, orient='index', columns=['Count'])
df_test = pd.DataFrame.from_dict(data_test, orient='index', columns=['Count'])

# Define Excel file names
excel_file_dev = 'dev_data.xlsx'
excel_file_train = 'train_data.xlsx'
excel_file_test = 'test_data.xlsx'

# Save dataframes to Excel files
with pd.ExcelWriter(excel_file_dev) as writer_dev:
    df_dev.to_excel(writer_dev, sheet_name='dev_data', index_label='Category')

with pd.ExcelWriter(excel_file_train) as writer_train:
    df_train.to_excel(writer_train, sheet_name='train_data', index_label='Category')

with pd.ExcelWriter(excel_file_test) as writer_test:
    df_test.to_excel(writer_test, sheet_name='test_data', index_label='Category')

print(f"Excel tables for 'dev', 'train', and 'test' data types have been created and saved.")
