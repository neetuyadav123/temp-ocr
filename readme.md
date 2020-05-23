# Devanagari-Character-Recognition

This is the final project of 7th semester BSC.CSIT.
<h1 align = 'center'>Introduction</h1>
Devanagari is the national font of Nepal and is used widely throughout the India also. It contains 10 numerals(०, १, २, ३, ४, ५, ६, ७, ८, ९) and 36 consonants (क, ख, ग, घ, ङ, च, छ, ज, झ, ञ, ट, ठ, ड, ढ, ण, त, थ, द, ध,न, प,फ, ब, भ, म, य, र, ल, व, श, ष, स, ह, क्ष, त्र, ज्ञ). Some consonants are complex and made by combine some other. However, throughout this project i considered them as single character.


## How to setup
* cd into project folder `cd <your-path-in-computer>/devnagiri_character_recognition`
* create a **virtual env** inside folder `python -m venv env`
* activate environment `source env/bin/activate` ; **For window user** env\Scripts\activate.bat
* Install dependencies. `pip install -r requirement.txt` This will take a while *get yourself a coffee*


# Run1 (require you to upload image)
`streamlit run app.py` *Open given link in your browser and =wait for 10 minutes= to finish loading_data process*


# Run2 (provide image input through camera)
`streamlit run video.py` *Open link in browser it will take input from camera*


# View notebook used while preparing this project (during training, etc)
`jupyter-notebook ai_notebooks` *if required open given url in browser and click on the notebook that you want to view*
**warning don't run any notebook cell if your computer is not strong enough**
