import os

def sort_files():
    directory = 'scraped_text'
    files = os.listdir(directory)
    files.sort()
    for index, filename in enumerate(files):

        file_extension = os.path.splitext(filename)[1]
        new_filename = f"{index}{file_extension}"

        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        os.rename(old_file_path, new_file_path)

