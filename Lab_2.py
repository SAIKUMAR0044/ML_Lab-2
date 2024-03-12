#1st question
import math

# Define a function named 'Euclidean_Distance' that calculates the Euclidean distance between two vectors
def Euclidean_Distance(Vector1, Vector2):
    # Check if the vectors have the same dimensions
    if len(Vector1) != len(Vector2):
        raise ValueError("Vectors must have the same dimensions")
    
    # Initialize the distance variable
    distance = 0
    
    # Iterate through each dimension of the vectors
    for i in range(len(Vector1)):
        # Sum the squared differences of corresponding elements
        distance += (Vector1[i] - Vector2[i]) ** 2
    
    # Take the square root of the sum to get the Euclidean distance
    return math.sqrt(distance)

# Define a function named 'Manhattan_Distance' that calculates the Manhattan distance between two vectors
def Manhattan_Distance(Vector1, Vector2):
    # Check if the vectors have the same dimensions
    if len(Vector1) != len(Vector2):
        raise ValueError("Vectors must have the same dimensions")
    
    # Initialize the distance variable
    distance = 0
    
    # Iterate through each dimension of the vectors
    for i in range(len(Vector1)):
        # Sum the absolute differences of corresponding elements
        distance += abs(Vector1[i] - Vector2[i])
    
    # Return the Manhattan distance
    return distance

# Example vectors
Vector1 = [1, 2, 3]
Vector2 = [4, 5, 6]

# Calculate and print the Euclidean distance between Vector1 and Vector2
euclidean_result = Euclidean_Distance(Vector1, Vector2)
print(f"The Euclidean distance between Vector1 and Vector2 is: {euclidean_result}")

# Calculate and print the Manhattan distance between Vector1 and Vector2
manhattan_result = Manhattan_Distance(Vector1, Vector2)
print(f"The Manhattan distance between Vector1 and Vector2 is: {manhattan_result}")

# Example usage:
Vector_a = [2, 8, 9]
Vector_b = [1, 4, 3]

euclidean_dist = Euclidean_Distance(Vector_a, Vector_b)
manhattan_dist = Manhattan_Distance(Vector_a, Vector_b)

print(f"Euclidean Distance: {euclidean_dist}")
print(f"Manhattan Distance: {manhattan_dist}")

#2nd question
import numpy as np
from collections import Counter

class KNN_Classifier:
    def __init__(self, k):
        # Constructor to initialize the KNN Classifier with a specified value of k
        self.k = k
    
    def fit(self, X_train, y_train):
        # Method to train the classifier with the training data
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        # Method to predict the labels for the test data
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Method to predict the label for a single data point using k-nearest neighbors
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Find the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Find the most common label among the k-nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage:
# Training data
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])

# Test data
X_test = np.array([[2.5, 3.5], [1.5, 2.5]])

# Create a KNN classifier with k=3
knn = KNN_Classifier(k=3)

# Train the classifier with the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
predictions = knn.predict(X_test)

# Print the predicted labels
print("Predictions:", predictions)

#3rd question
def label_encoding(categories):
    # Create a set of unique categories to obtain distinct labels
    unique_categories = set(categories)
    
    # Initialize an empty dictionary to store the mapping of categories to labels
    label_map = {}

    # Iterate over unique categories and assign a label (index) to each category
    for i, category in enumerate(unique_categories):
        label_map[category] = i

    # Return the resulting label mapping
    return label_map

# Example usage:
categories = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']

# Call the label_encoding function with the provided list of categories
label_map = label_encoding(categories)

# Print the resulting label encoding mapping
print("Label encoding:", label_map)

#4th question
def one_hot_encoding(categories):
    # Create a sorted set of unique categories to determine the order of one-hot encoding
    unique_categories = sorted(set(categories))
    
    # Initialize an empty list to store one-hot encoded vectors
    encoding = []

    # Iterate through each category in the input list
    for category in categories:
        # Create a one-hot vector with all zeros
        one_hot_vector = [0] * len(unique_categories)
        
        # Find the index of the current category in the unique categories list
        index = unique_categories.index(category)
        
        # Set the corresponding element in the one-hot vector to 1
        one_hot_vector[index] = 1
        
        # Append the one-hot vector to the encoding list
        encoding.append(one_hot_vector)

    # Return the resulting one-hot encoding
    return encoding

# Example usage:
categories = ['red', 'blue', 'green', 'red', 'green', 'blue']

# Call the one_hot_encoding function with the provided list of categories
one_hot_encoded = one_hot_encoding(categories)

# Print the resulting one-hot encoded vectors
print("One-Hot Encoded:")
for category, one_hot_vector in zip(categories, one_hot_encoded):
    print(category, "->", one_hot_vector)
