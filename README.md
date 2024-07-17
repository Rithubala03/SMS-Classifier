DESCRIPION:
          This repository contains code for a neural network-based SMS spam classifier using TensorFlow/Keras.The model is trained to distinguish between spam and ham (non-spam) SMS messages.It utilizes text preprocessing techniques such as tokenization and padding, followed by an embedding layer, global average pooling, and dense layers to perform binary classification. The dataset used is the SMS Spam Collection Dataset.

REQUIREMENTS:
               1)Python 3.x 
               2)TensorFlow Nightly: pip install tf-nightly 
               3)Pandas: pip install pandas 
               4)TensorFlow Datasets: pip install tensorflow-datasets 
               5)NumPy: pip install numpy 
               6)Matplotlib: pip install matplotlib

STEPS:

Step-1: Install Dependencies:Ensure you have Python 3.x installed. Install the required libraries: TensorFlow Nightly, Pandas, TensorFlow Datasets, NumPy, and Matplotlib.

Step-2: Download the Dataset: Download the training and validation datasets from the provided URLs.

Step-3: Load and Preprocess the Data: i)Load the datasets into Pandas DataFrames. ii)Encode the labels by mapping 'ham' to 0 and 'spam' to 1. iii)Tokenize the SMS messages using Keras' Tokenizer. iv)Pad the tokenized sequences to ensure uniform input length.

Step-4: Define the Model: i)Create a neural network model with an embedding layer to convert text into dense vectors. ii)Add a global average pooling layer to reduce the dimensions of the data. iii)Add dense layers with ReLU and sigmoid activations for classification.

Step-5: Train the Model: i)Compile the model using binary cross-entropy loss and the Adam optimizer. ii)Train the model using the padded training data and validate it with the test data.

Step-6: Predict Messages: i)Define a function to predict whether a given SMS message is spam or ham. ii)The function should tokenize and pad the input message, then use the trained model to make a prediction.

Step-7: Test Predictions: i)Test the model with predefined messages to ensure its accuracy. ii)Compare the model's predictions with the expected results to verify its performance.

USAGE: 
    Train the model with labeled SMS data,then deploy it to classify incoming messages as either spam or legitimate,ensuring robust spam detection in real-time applications.
