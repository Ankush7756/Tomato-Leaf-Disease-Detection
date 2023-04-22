Tomato Leaf Disease Detection using Convolutional Neural Networks
This project aims to detect tomato leaf diseases using Convolutional Neural Networks (CNNs). The dataset used for training and testing consists of images of tomato leaves with four types of diseases: Bacterial Spot, Early Blight, Late Blight, and Leaf Mold, as well as a healthy category.

Prerequisites
To run this project, you will need the following dependencies:

Python 3.x
TensorFlow 2.x
Keras
NumPy
Pandas
Matplotlib
OpenCV
You can install all the necessary dependencies using the following command:

Copy code
pip install -r requirements.txt
Dataset
The dataset used in this project is available on Kaggle at the following link:

https://www.kaggle.com/noulam/tomato

It consists of 18000 images of tomato leaves, with each category having 3000 images.

Running the Code
To train the model, run the following command:

Copy code
python train.py
To test the model, run the following command:

Copy code
python test.py
To predict the disease of a new image, run the following command:

css
Copy code
python predict.py --image_path path/to/image
Results
The trained model achieves an accuracy of 96% on the test set.

Future Work
In future, this project can be extended to detect tomato diseases in real-time using a camera, and to detect diseases in other crops as well.
