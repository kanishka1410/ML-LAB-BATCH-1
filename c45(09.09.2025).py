import math
from collections import Counter

def entropy(data):
    labels = [row[-1] for row in data]
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def split_dataset(data, index, value):
    subset = []
    for row in data:
        if row[index] == value:
            reduced_row = row[:index] + row[index+1:]
            subset.append(reduced_row)
    return subset

def gain_ratio(data, index):
    base_entropy = entropy(data)
    values = set(row[index] for row in data)

    new_entropy = 0.0
    split_info = 0.0
    for v in values:
        subset = split_dataset(data, index, v)
        prob = len(subset) / len(data)
        new_entropy += prob * entropy(subset)
        split_info -= prob * math.log2(prob) if prob > 0 else 0

    info_gain = base_entropy - new_entropy
    if split_info == 0:
        return 0
    return info_gain / split_info

def choose_best_feature(data):
    n_features = len(data[0]) - 1
    best_ratio, best_index = -1, -1
    for i in range(n_features):
        ratio = gain_ratio(data, i)
        if ratio > best_ratio:
            best_ratio, best_index = ratio, i
    return best_index

def majority_class(data):
    labels = [row[-1] for row in data]
    return Counter(labels).most_common(1)[0][0]

def build_tree(data, features):
    labels = [row[-1] for row in data]

    # If all labels are the same, return the label
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    # If no features left, return the majority class
    if len(data[0]) == 1:
        return majority_class(data)

    best_feat = choose_best_feature(data)
    best_feat_name = features[best_feat]
    tree = {best_feat_name: {}}
    feat_values = set(row[best_feat] for row in data)
    for value in feat_values:
        sub_features = features[:best_feat] + features[best_feat+1:]
        subset = split_dataset(data, best_feat, value)
        tree[best_feat_name][value] = build_tree(subset, sub_features)

    return tree

# Example dataset
dataset2 = [
    ['<=30', 'High', 'No', 'Fair', 'No'],
    ['<=30', 'High', 'No', 'Excellent', 'No'],
    ['31-40', 'High', 'No', 'Fair', 'Yes'],
    ['>40', 'Medium', 'No', 'Fair', 'Yes'],
    ['>40', 'Low', 'Yes', 'Fair', 'Yes'],
    ['>40', 'Low', 'Yes', 'Excellent', 'No'],
    ['31-40', 'Low', 'Yes', 'Excellent', 'Yes'],
    ['<=30', 'Medium', 'No', 'Fair', 'No'],
    ['<=30', 'Low', 'Yes', 'Fair', 'Yes'],
    ['>40', 'Medium', 'Yes', 'Fair', 'Yes'],
    ['<=30', 'Medium', 'Yes', 'Excellent', 'Yes'],
    ['31-40', 'Medium', 'No', 'Excellent', 'Yes'],
    ['31-40', 'High', 'Yes', 'Fair', 'Yes'],
    ['>40', 'Medium', 'No', 'Excellent', 'No']
]

features2 = ['Age', 'Income', 'Student', 'Credit']

c45_tree2 = build_tree(dataset2, features2)
print("C4.5 Decision Tree:", c45_tree2)
