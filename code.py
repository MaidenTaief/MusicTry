import json
import os


# Path to your .ipynb file
notebook_path = '/Users/taief/Desktop/MusicTry/Data_by_artist.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = json.load(f)

# Path to the output file
output_path = '/Users/taief/Desktop/MusicTry/Code_artist.py'


with open(output_path, 'w', encoding='utf-8') as output_file:
    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'code':
            # Write the code to the output file
            code = ''.join(cell['source'])
            output_file.write(code)
            output_file.write("\n# " + "-" * 40 + "\n")