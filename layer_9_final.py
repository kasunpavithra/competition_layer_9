#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, KFold


# In[2]:


"""Takes the text as input and save it  in the file specified in the implementation"""
def append_to_file(text):
    with open("outputs_optimized_layer_9_label_4.txt", "a") as file:
        # Write content to the file
        file.write(f"{text}\n")


# In[3]:


"""Returns the preprocessed train and valid sets without doing PCA"""
def get_preprocessed_except_pca(label, train, valid, append_file = False, testSet = False):

    dropping_labels = ["label_1", "label_2","label_3", "label_4"]
    # other labels should drop
    dropping_labels.remove(label)
    print(f"Running for {label} ")

    train.drop(dropping_labels, axis=1, inplace=True)
    
    if not testSet:
        valid.drop(dropping_labels, axis=1, inplace=True)

    if(len(train.columns[train.isnull().any()])>0):
        print(f"{label} has missing values in train set")
        train.dropna(inplace=True)

    if(len(valid.columns[valid.isnull().any()])>0):
        print(f"{label} has missing values in valid set")
        valid.dropna(inplace=True)

    # splitting features and the label
    x_train = train.drop([label], axis=1)
    y_train = train[label]
    
    if not testSet:
        x_valid = valid.drop([label], axis=1)
        y_valid = valid[label]
    else:
        x_valid = valid.drop(['ID'], axis=1)

    # print nessasary stuff
    print(f"initial train set shape={x_train.shape}")
    if append_file:
        append_to_file(f"initial train set shape={x_train.shape}")

    # initiate over sampling strategy
    smote = SMOTE(sampling_strategy='auto', random_state=42)  # You can adjust the sampling strategy

    # Fit and transform the dataset
    # rx_train, ry_train = smote.fit_resample(x_train, y_train)
    rx_train, ry_train = x_train, y_train

    # print after oversampling stuff
    print(f"Resampled train set shape={rx_train.shape}")
    if append_file:
        append_to_file(f"Resampled train set shape={rx_train.shape}")

    # init the scaler
    scaler = StandardScaler()
    

    # fit the scaler
    sx_train = pd.DataFrame(scaler.fit_transform(rx_train), columns=rx_train.columns)
    sx_valid = pd.DataFrame(scaler.transform(x_valid), columns=x_valid.columns)
    
    if not testSet:
        return sx_train, sx_valid, ry_train, y_valid
    else:
        return sx_train, sx_valid, ry_train


# In[4]:


"""Does pca"""
def do_pca(sx_train, sx_valid, n_comp = None):
    if n_comp is not None:
        pca = PCA(n_components= n_comp)

        psx_train = pca.fit_transform(sx_train)
        psx_valid = pca.transform(sx_valid)
        
        new_len = len(psx_train[0])
        
        psx_train = pd.DataFrame(psx_train, columns=[f"new_label{i}" for i in range(1, len(psx_train[0])+1)])
        psx_valid = pd.DataFrame(psx_valid, columns=[f"new_label{i}" for i in range(1, len(psx_valid[0])+1)])
    else:
        psx_train = sx_train
        psx_valid = sx_valid
    return psx_train, psx_valid


# In[5]:


"""Find optimal hyper parameters using random search and save outputs to the above mentioned text file"""
def find_optimal_hyper_paras(label, train, valid):
    sx_train, sx_valid, ry_train, y_valid = get_preprocessed_except_pca(label="label_1",train=train, valid=valid, append_to_file=True)
    for n_comp in [0.97, 0.98, 0.99, None]:
        print(f"Running for n_component{n_comp} for {label}")
        psx_train, psx_valid = do_pca(sx_train, sx_valid, n_comp=n_comp)
        print(f"No of new coums is {len(psx_train.columns)}.")
        append_to_file(f"No of new coums is {len(psx_train.columns)}.")
        # Create an instance of MyModel
    #         init_model = SVC()

    #         # Fit the model to the training data
    #         init_model.fit(x_train, y_train)

    #         # Make predictions on the test data
    #         y_pred = init_model.predict(x_valid)

    #         # Print the accuracy of the model
    #         accuracy = (y_pred == y_valid).mean()
    #         print(f"Accuracy for {label} with n_comp {n_comp}: {accuracy}")
    #         append_to_file(f"Initial accuracy for {label} with n_comp {n_comp}: {accuracy}")

        # Example of using RandomizedSearchCV to tune hyperparameters
        param_dist = {
            'C': [i for i in range(90,105)],
            'kernel': ['linear', 'rbf'],
            'gamma': uniform(0.0009, 0.1),
            "class_weight": ["balanced"]
        }

        svc = SVC()

        random_search = RandomizedSearchCV(
            estimator=svc,
            param_distributions=param_dist,
            n_iter=15,  # Number of random combinations to try
            cv=5,  # Number of cross-validation folds
            verbose=2,
            random_state=42,  # Set a random seed for reproducibility
            n_jobs=-1  # Use all available CPU cores for parallel computation
        )

        full_x = pd.concat([psx_train,psx_valid], axis = 0)
        full_y = pd.concat([ry_train, y_valid], axis = 0)

        random_search.fit(full_x, full_y)

        print(f"Best hyperparameters found by RandomizedSearchCV for label {label} with n_comp {n_comp}:")
        print(random_search.best_params_)
        append_to_file(f"Best params for {label} with n_comp {n_comp}: {random_search.best_params_}")

        print(f"Best Score: for label {label} with n_comp {n_comp}", random_search.best_score_)
        append_to_file(f"Best score for {label} with n_comp {n_comp}: {random_search.best_score_}")


# In[ ]:


#
# Run only when need know optimal hyper parameters
#
all_labels = ["label_1", "label_2","label_3", "label_4"]
for label in all_labels:
    find_optimal_hyper_paras(label, train = pd.read_csv("./train.csv"), valid =pd.read_csv("./valid.csv"))


# In[6]:


"""Evaluates the model with cross validation"""
def evaluate_model(label, n_comp = 0.98, kernal="rbf", gamma=0.001, C=100, class_weight= "balanced"):
    
    train=pd.read_csv("./train.csv")
    valid=pd.read_csv("./valid.csv")
    
    sx_train, sx_valid, ry_train, y_valid = get_preprocessed_except_pca(label=label, train=train, valid=valid, append_file=False)
    psx_train, psx_valid = do_pca(sx_train, sx_valid, n_comp=n_comp)
    
    full_x = pd.concat([psx_train,psx_valid], axis = 0)
    full_y = pd.concat([ry_train, y_valid], axis = 0)
    
    # Convert the DataFrames to NumPy arrays
    X = full_x.to_numpy()
    y = full_y.to_numpy().ravel()  # Flatten the labels to a 1D array

    # Specify the number of folds for cross-validation (k=5)
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize an SVM classifier (you can specify the kernel and other hyperparameters)
    classifier = SVC(kernel=kernal, gamma=gamma, C=C, class_weight=class_weight)  # You can change the kernel type as needed

    # Perform k-fold cross-validation
    cross_val_scores = cross_val_score(classifier, X, y, cv=kf)

    # Print the cross-validation scores
    print("Cross-validation scores:", cross_val_scores)

    # Calculate and print the mean and standard deviation of the cross-validation scores
    mean_score = np.mean(cross_val_scores)
    std_deviation = np.std(cross_val_scores)
    print("Mean accuracy:", mean_score)
    print("Standard deviation of accuracy:", std_deviation)


# In[8]:


evaluate_model(label="label_1", n_comp = 0.99, kernal="rbf", gamma=0.001, C=100, class_weight= "balanced")


# In[9]:


evaluate_model(label="label_1", n_comp = 0.99, kernal="rbf", gamma=0.001464, C=95.0, class_weight= "balanced")


# In[13]:


evaluate_model(label="label_1", n_comp = 0.99, kernal="rbf", gamma=0.0007, C=100, class_weight= "balanced")


# In[ ]:


evaluate_model(label="label_2", n_comp = 0.98, kernal="rbf", gamma=0.0009, C=100, class_weight= "balanced")
evaluate_model(label="label_3", n_comp = 0.97, kernal="rbf", gamma=0.0009, C=100, class_weight= "balanced")
evaluate_model(label="label_4", n_comp = 0.98, kernal="rbf", gamma=0.0009, C=100, class_weight= "balanced")


# In[14]:


"Train and predict the outputs for a label"
def train_and_predict_test(label, n_comp = 0.98, kernal="rbf", gamma=0.001, C=100, class_weight= "balanced"):
    train = pd.concat([pd.read_csv("./train.csv"),pd.read_csv("./valid.csv")], axis=0)
    test = pd.read_csv("./test.csv")
    
    sx_train, sx_test, ry_train = get_preprocessed_except_pca(label=label,train=train, valid=test, append_file=False, testSet=True)
    psx_train, psx_test = do_pca(sx_train, sx_test, n_comp=n_comp)
    
    classifier = SVC(kernel=kernal, gamma=gamma, C=C, class_weight=class_weight)
    
    classifier.fit(psx_train, ry_train)
    
    result = classifier.predict(psx_test)
    
    res_df = pd.DataFrame(result, columns=[label])
#     res_df.to_csv(f"./190438H_{label}.csv")
    return res_df


# In[44]:


train_and_predict_test(label="label_1", n_comp = 0.98, kernal="rbf", gamma=0.001, C=100, class_weight= "balanced")


# In[71]:


# values taken from the outputs_optimized_layer_10_label_4.txt file that is generated by running find_optimal_hyper_paras() method
config = [
    {"label": "label_1","n_comp": None, "kernal":"rbf", "gamma":0.001464, "C":95, "class_weight":"balanced" },
    {"label": "label_2","n_comp": None, "kernal":"rbf", "gamma":0.001464, "C":95, "class_weight":"balanced" },
    {"label": "label_3","n_comp": None, "kernal":"rbf", "gamma":0.001464, "C":95, "class_weight":"balanced" },
    {"label": "label_4","n_comp": None, "kernal":"rbf", "gamma":0.001464, "C":95, "class_weight":"balanced" },
]


# In[72]:


dfs = []
for row in config:
    dfs.append(train_and_predict_test(label=row["label"], n_comp=row["n_comp"], kernal=row["kernal"], gamma=row["gamma"], C=row["C"], class_weight=row["class_weight"]))
final = pd.concat(dfs, axis=1)
final["ID"] = [i for i in range(1, final.shape[0]+1)]
final.to_csv("190438H_layer_9_att_3.csv", index=False)


# In[15]:


# values taken from the outputs_optimized_layer_10_label_4.txt file that is generated by running find_optimal_hyper_paras() method
config = [
    {"label": "label_1","n_comp": 0.99, "kernal":"rbf", "gamma":0.0009, "C":100.0, "class_weight":"balanced" },
    {"label": "label_2","n_comp": 0.98, "kernal":"rbf", "gamma":0.001, "C":100.0, "class_weight":"balanced" },
    {"label": "label_3","n_comp": 0.97, "kernal":"rbf", "gamma":0.001, "C":100.0, "class_weight":"balanced" },
    {"label": "label_4","n_comp": 0.98, "kernal":"rbf", "gamma":0.001, "C":100.0, "class_weight":"balanced" },
]


# In[16]:


dfs = []
for row in config:
    dfs.append(train_and_predict_test(label=row["label"], n_comp=row["n_comp"], kernal=row["kernal"], gamma=row["gamma"], C=row["C"], class_weight=row["class_weight"]))
final = pd.concat(dfs, axis=1)
final["ID"] = [i for i in range(1, final.shape[0]+1)]
final.to_csv("190438H_layer_9_att_5.csv", index=False)


# In[ ]:




