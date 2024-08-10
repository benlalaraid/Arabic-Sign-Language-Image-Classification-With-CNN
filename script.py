import os

def list_subdirectories(directory):
    subdirectories = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            subdirectories.append(item)
    return subdirectories
main_directory = './data'
subdirs = list_subdirectories(main_directory)
print(subdirs)
