#1st question
import math

def Euclidean_Distance(Vector1, Vector2):
    if len(Vector1) != len(Vector2):
        raise ValueError("Vectors must have the same dimensions")
    
    distance = 0
    for i in range(len(Vector1)):
        distance += (Vector1[i] - Vector2[i]) ** 2
    
    return math.sqrt(distance)

def Manhattan_Distance(Vector1, Vector2):
    if len(Vector1) != len(Vector2):
        raise ValueError("Vectors must have the same dimensions")
    
    distance = 0
    for i in range(len(Vector1)):
        distance += abs(Vector1[i] - Vector2[i])
    
    return distance

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
    def __init__(self,k):
        self.k=k
    def fit(self, X_train, y_train):
        self.X_train=X_train
        self.y_train=y_train
    def predict(self, X_test):
        y_pred=[self._predict(x) for x in X_test]
        return np.array(y_pred)
    def _predict(self,x):
        distances=[np.linalg.norm(x-x_train) for x_train in self.X_train]
        k_indices=np.argsort(distances[:self.k])
        k_nearest_labels=[self.y_train[i] for i in k_indices]
        most_common=Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
X_train=np.array([[1,2],[2,3],[3,4],[4,5]])
y_train=np.array([0,0,1,1])
X_test=np.array([[2.5,3.5],[1.5,2.5]])
knn=KNN_Classifier(k=3)
knn.fit(X_train,y_train)
predictions=knn.predict(X_test)
print("predictions:",predictions)

#3rd question
def label_encoding(cateogries):
    unique_cateogries=set(cateogries)
    label_map={}

    for i, cateogries in enumerate(unique_cateogries):
        label_map[cateogries]=i

    return label_map
cateogries=['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
label_map=label_encoding(cateogries)
print("label encoding:",label_map)

#4th question
def one_hot_encoding(cateogries):
    unique_cateogries=sorted(set(cateogries))
    encoding=[]
    for cateogry in cateogries:
        one_hot_vector=[0]*len(unique_cateogries)
        index=unique_cateogries.index(cateogry)
        one_hot_vector[index]=1
        encoding.append(one_hot_vector)

    return encoding
categories = ['red', 'blue', 'green', 'red', 'green', 'blue']

one_hot_encoded = one_hot_encoding(categories)
print("One-Hot Encoded:")
for category, one_hot_vector in zip(categories, one_hot_encoded):
    print(category, "->", one_hot_vector)
