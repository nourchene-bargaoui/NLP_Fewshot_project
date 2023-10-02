import os
import sys


# Function to get all files recursively from a folder
def get_files_recursively(folder_path):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

# Function to pick files with a specific extension (e.g., ".ann") from a list
def pick_ann(given_list: list):
    lis = []
    for i in given_list:
        if i.endswith(".ann"):
            lis.append(i)
    return lis

# Function to read lines from a file into a list, with optional filtering by lines starting with a specific string
def read_file_lines_to_list(file_path, ignore_starting_with=None, ignore_error=True):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        if ignore_starting_with is not None:
            lines = [line for line in lines if not line.startswith(ignore_starting_with)]
        return lines
    except FileNotFoundError:
        if not ignore_error:
            print(f"File not found: {file_path}")
        return []
    except Exception as e:
        if not ignore_error:
            print(f"An error occurred: {e}")
        return []

# Function to combine multiple lists into a single list
def combine_lists(*args):
    combined_list = []
    for lst in args:
        combined_list.extend(lst)
    return combined_list

# Function to count entities in data
def count_entities(data):
    entity_count = {}
    relationship_count = {}
    for row in data:
        columns = row.split()
        entity_type = columns[1]  # Get the second column which contains the entity type
        if columns[0].startswith("T"):  # Entity
            entity_count[entity_type] = entity_count.get(entity_type, 0) + 1
        elif columns[0].startswith("R"):  # Relationship
            relation_type = columns[0].split("-")[0]
            if relation_type not in relationship_count:
                relationship_count[relation_type] = 1
            else:
                relationship_count[relation_type] += 1
    return entity_count

def main():

    folders = []
    if (len(sys.argv) == 1):
        # List of folders to process
        folders = [r'F:\Repo\PycharmProjects\NLP_Fewshot_project\Entity\dev', r'F:\Repo\PycharmProjects\NLP_Fewshot_project\Entity\train', r'F:\Repo\PycharmProjects\NLP_Fewshot_project\Entity\test']
    else:
        folders = [i.replace(" ","") for i in sys.argv[1].split(",")]

    # Initialize a list to store all the strings
    all_string = []

    # Process each folder
    for j in folders:
        for i in pick_ann(get_files_recursively(j)):
            # Combine all strings from the selected files
            all_string = combine_lists(all_string, read_file_lines_to_list(i, "R"))
        print("data Type: {}".format(j.split("\\")[-1]))
        # Count entities in the combined strings
        print(count_entities(all_string))
        print()

main()
