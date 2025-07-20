#### Potato-Disease-Prediction
Potato-Disease-Prediction is a full-stack web application for detecting potato leaf diseases using Convolutional Neural Networks (CNN). The system includes a FastAPI backend for model inference and a ReactJS frontend for a user-friendly interface. Users can upload images of potato leaves to get real-time predictions for common diseases such as Early Blight, Late Blight, and Healthy.

### Training the Model
Follow these steps to train the model:

1. Download Dataset from Kaggle
  Visit [Kaggle - PlantVillage Dataset]
2. Download the dataset and extract only the folders related to potatoes.
3. Move those folders into:
   Training/Data/
    
4.Start Jupyter Notebook
   If you don’t have Jupyter installed:
     <pre>```bash pip install notebook```</pre>

5.To run Jupyter in browser:
    <pre> ```jupyter notebook``` </pre>
   This will open Jupyter in your default browser.

6.Train the Model
   Open Training/model_training.ipynb in Jupyter Notebook.
   Run each cell step-by-step to
   Copy the model generated and save it with the version number in the models folder.

### Running the API
Using FastAPI
  1. Get inside api folder
    <pre> ```cd api``` </pre>
  2. Run the FastAPI Server using uvicorn
    <pre>```uvicorn main:app --reload --host 0.0.0.0```</pre>
Your API is now running at 0.0.0.0:8001

### Using FastAPI & TF Serve
  1. Get inside api folder
    <pre>```cd api```</pre>
  2. Copy the models.config.example as models.config and update the paths in file.
  3.  Run the TF Serve (Update config file path below)
     <pre>``` docker run -t --rm -p 8000:8000 -v "C:/My Programs/Machine Learning Projects/potato-disease-classification:/potato-disease-classification" tensorflow/serving  --rest_api_port=8000 --model_config_file=/potato-disease-classification/models.config```</pre>
 4. Run the FastAPI Server using uvicorn For this you can directly run it from your  main-tf-serving.py
 5. Your API is now running at 0.0.0.0:8000

### Running the Frontend
1.Get inside frontend folder
      <pre>```cd frontend```</pre>
2.Copy the .env.example as .env and update REACT_APP_API_URL to API URL if needed.
3.Run the frontend
     <pre>```npm run start```</pre>
  
