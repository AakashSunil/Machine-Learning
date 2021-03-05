import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Menu to choose the type of tree
def menu():
    validation_data_choice = None
    menu_dict = {1:'With Entropy as the Impurity', 2:'With Variance as the Impurity', 3:'With Entropy as the Impurity and Reduced-Error-Pruning', 
    4:'With Variance as the Impurity and Reduced-Error-Pruning', 5:'With Entropy as the Impurity and Depth-Based-Pruning', 6:'With Variance as the Impurity and Depth-Based-Pruning',
    7:'Random Forest Classifier'}
    for option in menu_dict:
        print(option,menu_dict[option])
    tree_choice = int(input("\nEnter the Option:"))
    print(menu_dict[tree_choice],"\n")
    training_data_choice,test_data_choice,validation_data_choice = datasetMenu(tree_choice)

    return tree_choice,training_data_choice,test_data_choice,validation_data_choice

# Menu to choose the Datasets - Train, Test and Validation
def datasetMenu(choice):
    print('Choose the Dataset from the Option Below\n')
    dataset_dict = {1:'c300_d100',2:'c300_d1000',3:'c300_d5000',
    4:'c500_d100',5:'c500_d1000',6:'c500_d5000',
    7:'c1000_d100',8:'c1000_d1000',9:'c1000_d5000',
    10:'c1500_d100',11:'c1500_d1000',12:'c1500_d5000',
    13:'c1800_d100',14:'c1800_d1000',15:'c1800_d5000'
    }
    for option in dataset_dict:
        print(option,dataset_dict[option])
    dataset_choice = int(input('Enter the Dataset Choice:'))
    training_data = './all_data/train_'+ dataset_dict[dataset_choice]+'.csv' 
    testing_data = './all_data/test_'+ dataset_dict[dataset_choice]+'.csv' 
    validation_data = './all_data/valid_'+ dataset_dict[dataset_choice]+'.csv' 
    return training_data,testing_data,validation_data
    
# Function to read the dataset files - Train, Test and if Required Vaildation Dataset
def readFiles(train_file,test_file,vaild_file,tree_type=1):
    train_data = None
    test_data = None
    valid_data = None
    print(train_file,test_file,vaild_file)
    # Reading the data into a csv
    train_data = pd.read_csv(train_file, header=None)

    # Renaming Columns for easier Understanding
    # Variables renamed as 1 -> Attr1, 2 -> Attr2...
    # Final Result Variable renamed to Label
    _,columns = train_data.shape
    column_list = []
    for i in range(columns-1):
        final_str = 'Attr'+str(i)
        column_list.append(final_str)
    column_list.append('label')
    train_data.columns = column_list

    # Reading the data into a csv
    test_data = pd.read_csv(test_file, header=None)
    # Renaming Columns for easier Understanding
    # Variables renamed as 1 -> Attr1, 2 -> Attr2...
    # Final Result Variable renamed to Label
    _,columns = train_data.shape
    column_list = []
    for i in range(columns-1):
        final_str = 'Attr'+str(i)
        column_list.append(final_str)
    column_list.append('label')
    test_data.columns = column_list

    if(tree_type > 2 and tree_type < 7):
        # Reading the data into a csv
        valid_data = pd.read_csv(train_file, header=None)
        # Renaming Columns for easier Understanding
        # Variables renamed as 1 -> Attr1, 2 -> Attr2...
        # Final Result Variable renamed to Label
        _,columns = train_data.shape
        column_list = []
        for i in range(columns-1):
            final_str = 'Attr'+str(i)
            column_list.append(final_str)
        column_list.append('label')
        valid_data.columns = column_list
    return train_data,test_data,valid_data

# Calculate Entropy
def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

# Calculate Overall Entropy of the Split Data
def calculate_overall_entropy(data_below,data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy

# Calculate Variance
def calculate_variance(data):
    label_column = data[:, -1]
    elements, counts = np.unique(label_column, return_counts=True)
    variance = np.product([counts[i]/np.sum(counts) for i in range(len(elements))])
    return variance

# Calculate Overall Variance of the Split Data
def calculate_overall_variance(data_below,data_above):

    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_variance =  (p_data_below * calculate_variance(data_below) 
                      + p_data_above * calculate_variance(data_above))
    
    return overall_variance

# Potential Splits where the data can be split to have better output - Data Points at mid location of the two splits
def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2
                
                potential_splits[column_index].append(potential_split)
    
    return potential_splits

# Splitting of the Data - to reduce the load on recursion. Individual branches data algorithm recursion 
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]
    
    return data_below, data_above

# Finding the best attribute to split the data on
def determine_best_split(data, potential_splits, tree_type):
    
    overall_entropy = 9999
    overall_variance = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            if(tree_type == 2) or (tree_type == 4) or (tree_type == 6):
                current_overall_variance = calculate_overall_variance(data_below, data_above)
                if current_overall_variance <= overall_variance:
                    overall_variance = current_overall_variance
                    best_split_column = column_index
                    best_split_value = value
            else:

                current_overall_entropy = calculate_overall_entropy(data_below, data_above)
                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value
    
    return best_split_column, best_split_value

# Function to Calculate the Information Gain with Respect to Entropy
def Information_Gain_Entropy(data,column_attribute,target_attribute):
    
    # Calculation of Entropy of the Class Column
    dataset_entropy = calculate_entropy(data[target_attribute])

    # Counts and values of the Column Attribute in play
    values,counts= np.unique(data[column_attribute],return_counts=True)
    
    # Summation of the Entropies of the unique values in the Column Attribute
    individual_entropy = np.sum([(counts[i]/np.sum(counts))*calculate_entropy(data.where(data[column_attribute]==values[i]).dropna()[target_attribute]) for i in range(len(values))])

    # Calculation of the Information Gain 
    information_gain = dataset_entropy - individual_entropy
    return information_gain

# Function to Calculate the Information Gain with Respect to Entropy
def Information_Gain_Variance(data,column_attribute,target_attribute):
    
    # Calculation of Entropy of the Class Column
    dataset_variance = calculate_variance(data[target_attribute])

    # Counts and values of the Column Attribute in play
    values,counts= np.unique(data[column_attribute],return_counts=True)
    
    # Summation of the Entropies of the unique values in the Coljumn Attribute
    individual_variance = np.sum([(counts[i]/np.sum(counts))*calculate_variance(data.where(data[column_attribute]==values[i]).dropna()[target_attribute]) for i in range(len(values))])

    # Calculation of the Information Gain 
    information_gain = dataset_variance - individual_variance
    return information_gain

# Random Forest Implementation
def randomForestClassifier(X_train,Y_train,X_test,Y_test):
    
    # sklearn Library - Default settings
    clf=RandomForestClassifier(n_estimators=100)
    # Training the Data
    clf.fit(X_train,Y_train)
    # Prediction of test data on the training data
    y_pred=clf.predict(X_test)
    # Accuracy Measure
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

# To check for a particular row - Prediction of the test data
def classify_example(example, tree):
    node = list(tree.keys())[0]
    feature_name, _, value = node.split()

    if example[feature_name] <= float(value):
        child = tree[node][0]
    else:
        child = tree[node][1]

    # Leaf Node Case
    if not isinstance(child, dict):
        return child
    
    # Recursion
    else:
        residual_tree = child
        return classify_example(example, residual_tree)

# To calculate the Accuracy of the tree for the dataset
def calculate_accuracy(df, tree):

    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["label"]
    
    accuracy = df["classification_correct"].mean()
    
    return accuracy

# To check purity of the data - Single Unique class in the set
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False

# Data Classification - To make the leaf node once the data in the set is Pure
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

# Decision Tree Algorithm without Pre-Pruning
def decision_tree_algorithm(df, tree_type,counter=0):
    
    # Data Preparations
    if counter == 0:
        global ATTRIBUTES
        ATTRIBUTES = df.columns
        data = df.values
    else:
        data = df           
    
    
    # Cases where leaf node is created
    if (check_purity(data)):
        classification = classify_data(data)
        return classification

    
    # Recursion
    else:    
        counter += 1

        # Gettinig the Potential Splits and the Splitting of data based on heuristic
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits, tree_type)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # Create the Sub-Tree
        feature_name = ATTRIBUTES[split_column]
        best_feature = "{} <= {}".format(feature_name,split_value)
        sub_tree = {best_feature: []}
        
        # Branches based on data - Left or Right
        left_child = decision_tree_algorithm_preprune(data_below, tree_type, counter)
        right_child = decision_tree_algorithm_preprune(data_above, tree_type, counter)
        
        # If both branches give same result - then splitting is not needed and can be reduced to a single node
        if left_child == right_child:
            sub_tree = left_child
        else:
            sub_tree[best_feature].append(left_child)
            sub_tree[best_feature].append(right_child)
        
        return sub_tree

# Decision tree algorithm with depth based pruning
def decision_tree_algorithm_preprune(df, tree_type, max_depth,counter=0,min_sample=2):
    # Data Preparations
    if counter == 0:
        global ATTRIBUTES
        ATTRIBUTES = df.columns
        data = df.values
    else:
        data = df           
    
    
    # Cases where leaf node is created
    if (check_purity(data)) or (counter == max_depth):
        classification = classify_data(data)
        return classification

    
    # Recursion
    else:    
        counter += 1

        # Gettinig the Potential Splits and the Splitting of data based on heuristic
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits, tree_type)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # Create the Sub-Tree
        feature_name = ATTRIBUTES[split_column]
        best_feature = "{} <= {}".format(feature_name,split_value)
        sub_tree = {best_feature: []}
        
        # Branches based on data - Left or Right
        left_child = decision_tree_algorithm_preprune(data_below, tree_type, max_depth,counter)
        right_child = decision_tree_algorithm_preprune(data_above, tree_type, max_depth,counter)
        
        # If both branches give same result - then splitting is not needed and can be reduced to a single node
        if left_child == right_child:
            sub_tree = left_child
        else:
            sub_tree[best_feature].append(left_child)
            sub_tree[best_feature].append(right_child)
        
        return sub_tree


# Post Pruning - To Improve Accuracy of the Decision Tree
def post_pruning(tree, df_train, df_val):
    
    node = list(tree.keys())[0]
    left_child, right_child = tree[node]

    # base case
    if not isinstance(left_child, dict) and not isinstance(right_child, dict):
        return pruning_result(tree, df_train, df_val)
        
    # recursive part
    else:
        df_train_left, df_train_right = filter_df(df_train, node)
        df_val_left, df_val_right = filter_df(df_val, node)
        
        if isinstance(left_child, dict):
            left_child = post_pruning(left_child, df_train_left, df_val_left)
            
        if isinstance(right_child, dict):
            right_child = post_pruning(right_child, df_train_right, df_val_right)
            
        tree = {node: [left_child, right_child]}
    
        return pruning_result(tree, df_train, df_val)

# Function to return Leaf node or Sub-Tree on making predictions using Validation Dataset
def pruning_result(tree, df_train, df_val):

    leaf = df_train.label.value_counts().index[0]
    errors_leaf = sum(df_val.label != leaf)
    errors_decision_node = sum(df_val.label != make_predictions(df_val, tree))
    if errors_leaf <= errors_decision_node:
        return leaf
    else:
        return tree

# Function to get prediction Values on the dataset on the tree (pruned tree)
def make_predictions(df, tree):
    
    if len(df) != 0:
        predictions = df.apply(predict_example, args=(tree,), axis=1)
    else:
        # "df.apply()"" with empty dataframe returns an empty dataframe,
        # but "predictions" should be a series instead
        predictions = pd.Series()
        
    return predictions

# Function to make the individual predicctions
def predict_example(example, tree):
    
    # tree is just a root node
    if not isinstance(tree, dict):
        return tree
    
    node = list(tree.keys())[0]
    feature_name, comparison_operator, value = node.split(" ")
    # ask node
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[node][0]
        else:
            answer = tree[node][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)

#filter out the positve and negative examples - Left and Right branch data
def filter_df(df, node):
    feature, _, value = node.split()
    df_yes = df[df[feature] <= float(value)]
    df_no  = df[df[feature] >  float(value)]
    
    return df_yes, df_no

# Driver Program
print('Welcome to Decision Tree Implementation\n\n')
tree_type,train_file,test_file,vaild_file = menu()
train_data,test_data,vaild_data = readFiles(train_file,test_file,vaild_file,tree_type)
# train_data,test_data,vaild_data = readFiles("train_c1500_d100.csv","test_c1500_d100.csv","valid_c1500_d100.csv")
max_depths = [5,10,15,20,50,100]



if tree_type in [1,2,3,4]:
    tree = decision_tree_algorithm(train_data,tree_type)
    # pprint(tree)
    accuracy = calculate_accuracy(test_data, tree)
    print('Pre-Pruned accuracy -->',accuracy)
    if tree_type in [3,4]:
        tree_pruned = post_pruning(tree,train_data,vaild_data)
        # pprint(tree_pruned)
        accuracy_post_prune = calculate_accuracy(test_data, tree_pruned)
        print('Post-Pruned Accuracy -->',accuracy_post_prune)

elif(tree_type == 5) or (tree_type == 6):
    for i in max_depths:
        tree = decision_tree_algorithm_preprune(train_data,tree_type,i)
        # pprint(tree)
        accuracy = calculate_accuracy(test_data, tree)
        print('Depth=',i,'Accuracy -->',accuracy)
elif(tree_type == 7):
    X_train = train_data.iloc[:,:-1]
    Y_train =  train_data.iloc[:,-1]
    X_test = test_data.iloc[:,:-1]
    Y_test = test_data.iloc[:,-1]
    randomForestClassifier(X_train,Y_train,X_test,Y_test)
else:
    print('Invalid Option')
