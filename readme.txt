1.Create Project Directory: 
Create the base project folder 
(project) and subdirectories like 
models and dataset using the following commands:
Inside dataset/, place your train/, val/, and test/ folders, which will contain the 
images for training, validation, and testing, respectively.
---------------------******--------------------------
Step 2: Install Dependencies
Create a requirements.txt file with the necessary dependencies. You can create it manually or generate it using pip freeze > 
requirements.txt. Here are the key libraries you need:
streamlit
tensorflow
numpy
pillow
pip install -r requirements.txt
Step 3: Model Training Script (create_model.py)
The training script will build and train the pneumonia detection model. It will 
save the model as pneumonia_model.h5 in the models/ folder.