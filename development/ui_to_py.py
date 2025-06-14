import os

# Script must be run in the same directory as the .ui file

anaconda_path = "/opt/anaconda3/bin" # Path to the anaconda3 bin directory
ui_file = "Main-GUIDesign.ui" # Name of the .ui file
output_name = "newGUI.py" # Name of the output .py file

os.system(f"{anaconda_path}/pyuic5 -x {ui_file} -o {output_name}")

if os.path.exists(output_name):
    print("Conversion successful")
else:
    print("Conversion failed... Check your paths...")