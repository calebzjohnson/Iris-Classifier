# Author: Caleb Johnson
# Date: 12/15/25
# Course: CS131 Artificial Intelligence
# Assignment: A6 Artificial Neural Networks

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DataProcessing:
    @staticmethod
    # return: training inputs, testing inputs, training labels, testing labels, label_encoder
    def process_data(filepath):
        data = data = np.genfromtxt(filepath, dtype=str, delimiter=',', autostrip=True)
        input_data = data[:, :-1] # trait data
        output_data = data[:, -1] # species name

        # Encode species names
        label_encoder = LabelEncoder()
        encoded_output = label_encoder.fit_transform(output_data)

        # Split into train/test and scale for better performance
        i_train, i_test, o_train, o_test = train_test_split(input_data, encoded_output, test_size=0.2, shuffle=True, random_state=42)
        scaler = StandardScaler()
        i_train_scaled = scaler.fit_transform(i_train)
        i_test_scaled = scaler.transform(i_test)

        return i_train_scaled, i_test_scaled, o_train, o_test, label_encoder, scaler

def user_predict(ntwk, encoder, scaler):
    sepal_l = float(input("Sepal Length(cm): "))
    sepal_w = float(input("Sepal Width(cm): "))
    petal_l = float(input("Petal Length(cm): "))
    petal_w = float(input("Petal Width(cm): "))

    user_input = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    input_scaled = scaler.transform(user_input)
    encoded_pred = ntwk.predict(input_scaled)
    decoded_pred = encoder.inverse_transform(encoded_pred)[0]

    print(f"Predicted Species: {decoded_pred}")

if __name__ == "__main__":
    data_processing = DataProcessing()
    train_input, test_input, train_label, test_label, label_encoder, scaler = data_processing.process_data('iris_data.txt')

    # define and train the mdoel
    ntwk = MLPClassifier(
        hidden_layer_sizes=(5, 5),
        max_iter=1000,
        alpha=0.001,
        learning_rate_init=0.01,
        random_state=42)
    ntwk.fit(train_input, train_label)
    test_predictions = ntwk.predict(test_input)
    accuracy = accuracy_score(test_label, test_predictions) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # get user input
    user_input = input("Do you want to input your own iris data (y/n)? ").strip().lower()
    if user_input == "y":
        user_predict(ntwk, label_encoder, scaler)
    
    print("Exiting program")