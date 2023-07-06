import numpy as np
import pandas as pd
import numpy as np
from somethingelse import ClassAssRules
from sklearn.metrics import accuracy_score

attribute_names = [
  "class_name", "handicapped-infants", "water-project-cost-sharing",
  "adoption-of-the-budget-resolution", "physician-fee-freeze",
  "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
  "aid-to-nicaraguan-contras", "mx-missile", "immigration",
  "synfuels-corporation-cutback", "education-spending",
  "superfund-right-to-sue", "crime", "duty-free-exports",
  "export-administration-act-south-africa"
]

attribute_names = [x.replace('-', '_') for x in attribute_names]

df2 = pd.read_csv("house-votes-84.data", names=attribute_names)

# Replace '?' with NaN
df2.replace('?', np.nan, inplace=True)

df2.dropna(inplace=True)

# Reset the index
df2.reset_index(drop=True, inplace=True)

# Drop the class_name column
df_without_class = df2.drop('class_name', axis=1, inplace=False)
df_encoded = pd.get_dummies(df_without_class)


target_class = df2['class_name'].tolist()

def cross_validation(data, labels, model, num_folds=10):
    fold_size = len(data) // num_folds
    accuracy_scores = []

    for fold in range(num_folds):
        # Split the data into training and validation sets
        validation_data = data[fold*fold_size:(fold+1)*fold_size]
        validation_labels = labels[fold*fold_size:(fold+1)*fold_size]
        train_data = np.concatenate([data[:fold*fold_size], data[(fold+1)*fold_size:]])
        train_labels = np.concatenate([labels[:fold*fold_size], labels[(fold+1)*fold_size:]])


        # Train the model on the training data
        #model.fit(train_data, train_labels)

        # Evaluate the model on the validation data
        #accuracy = model.evaluate(validation_data, validation_labels)

        #accuracy_scores.append(accuracy)

    #mean_accuracy = np.mean(accuracy_scores)
    #return mean_accuracy
cross_validation(df_encoded, target_class, None)