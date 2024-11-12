from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics

main = tkinter.Tk()
main.title("Machine Learning and Regression Approach for Predicting the Right Group of Customers for Automobile Industries") 
main.geometry("1000x650")

global filename
global x_train,y_train,x_test,y_test
global X, Y
global le
global dataset
accuracy = []
precision = []
recall = []
fscore = []
global classifier
global cnn_model

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n\n")

def preprocessDataset():
    global X,y
    global le
    global dataset
    global x_train,y_train,x_test,y_test
    le = LabelEncoder()
    text.delete('1.0', END)
    dataset.dropna(inplace = True)
    print(dataset.info())
    text.insert(END,str(dataset.head())+"\n\n")
    
    # Create a count plot
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    # Replace 'dataset' with your actual DataFrame and 'Drug' with the column name
    ax = sns.countplot(x='Segmentation', data=dataset, palette="Set3")
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Annotate each bar with its count value
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

    plt.show()  # Display the plot
    le = LabelEncoder()
    dataset['Gender']=le.fit_transform(dataset['Gender'])
    dataset['Ever_Married']=le.fit_transform(dataset['Ever_Married'])
    dataset['Graduated']=le.fit_transform(dataset['Graduated'])
    dataset['Profession']=le.fit_transform(dataset['Profession'])
    dataset['Spending_Score']=le.fit_transform(dataset['Spending_Score'])
    dataset['Var_1']=le.fit_transform(dataset['Var_1'])
    X=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values
    text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    text.insert(END,"Total records found in dataset to train: "+str(x_train.shape[0])+"\n\n")
    text.insert(END,"Total records found in dataset to test: "+str(x_test.shape[0])+"\n\n")
    print(x_train)

def analysis():
    # Set up subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    sns.countplot(x='Gender', data=dataset, ax=axes[0, 0])
    sns.countplot(x='Ever_Married', data=dataset, ax=axes[0, 1])
    sns.countplot(x='Graduated', data=dataset, ax=axes[0, 2])
    sns.countplot(x='Spending_Score', data=dataset, ax=axes[1, 0])
    sns.countplot(x='Var_1', data=dataset, ax=axes[1, 1])

    # Histogram
    sns.histplot(dataset['Age'], bins=30, kde=True, ax=axes[1, 2])

    # Show the plots
    plt.tight_layout()
    plt.show()


def custom_knn_classifier():
    global x_train, y_train
    model_file_path = "knn_model.pkl"
    
    # Check if the model file exists
    if os.path.exists(model_file_path):
        # If the model file exists, load the model
        KNN = joblib.load(model_file_path)
    else:
        # If the model file doesn't exist, train the model and save it
        KNN = KNeighborsClassifier(n_neighbors=10, leaf_size=30, metric='minkowski')
        KNN.fit(X_train,y_train)
        # Save the model
        joblib.dump(KNN, model_file_path)
    
    # Now use the trained or loaded model for predictions
    predict = KNN.predict(x_test)
    
    # The rest of your code for evaluation and reporting
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, "KNN Precision : " + str(p) + "\n")
    text.insert(END, "KNN Recall    : " + str(r) + "\n")
    text.insert(END, "KNN FMeasure  : " + str(f) + "\n")
    text.insert(END, "KNN Accuracy  : " + str(a) + "\n\n")
    # Update the confusion matrix and classification report code
    cm = confusion_matrix(y_test, predict)
    report = classification_report(y_test, predict)
    
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)

def Randomforestclassifier():
    global x_train, y_train
    model_file_path1 = "RF_model.pkl"
    
    # Check if the model file exists
    if os.path.exists(model_file_path1):
        # If the model file exists, load the model
        RF = joblib.load(model_file_path1)
    else:
        # If the model file doesn't exist, train the model and save it
        RF = RandomForestClassifier()
        RF.fit(X_train,y_train)
        # Save the model
        joblib.dump(RF, model_file_path1)
    
    # Now use the trained or loaded model for predictions
    predict = RF.predict(x_test)
    
    # The rest of your code for evaluation and reporting
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, "RF Precision : " + str(p) + "\n")
    text.insert(END, "RF Recall    : " + str(r) + "\n")
    text.insert(END, "RF FMeasure  : " + str(f) + "\n")
    text.insert(END, "RF Accuracy  : " + str(a) + "\n\n")
    
    # Update the confusion matrix and classification report code
    cm = confusion_matrix(y_test, predict)
    report = classification_report(y_test, predict)
    
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)

def Prediction():
    model_file_path1 = "RF_model.pkl"
    
    # Check if the model file exists
    if os.path.exists(model_file_path1):
        # If the model file exists, load the model
        RF = joblib.load(model_file_path1)
    else:
        # If the model file doesn't exist, train the model and save it
        RF = RandomForestClassifier()
        RF.fit(X_train,y_train)
    
    #prediction
    predictions=RF.predict(x_test)
    # Display predictions and input data in the Tkinter Text widget
    text.insert(END, "Sample\t\t\tPrediction\n")
    text.insert(END, "-"*50 + "\n")
    for i in range(len(predictions)):
        text.insert(END, f"{x_test[i]} Classified as **********************{predictions[i]}\n")   
    
def graph():
    # Create a DataFrame
    df = pd.DataFrame([
    ['KNN', 'Precision', precision[0]],
    ['KNN', 'Recall', recall[0]],
    ['KNN', 'F1 Score', fscore[0]],
    ['KNN', 'Accuracy', accuracy[0]],
    ['rf', 'Precision', precision[-1]],
    ['rf', 'Recall', recall[-1]],
    ['rf', 'F1 Score', fscore[-1]],
    ['rf', 'Accuracy', accuracy[-1]],
    ], columns=['Parameters', 'Algorithms', 'Value'])

    # Pivot the DataFrame and plot the graph
    pivot_df = df.pivot_table(index='Parameters', columns='Algorithms', values='Value', aggfunc='first')
    pivot_df.plot(kind='bar')
    # Set graph properties
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Display the graph
    plt.show()
def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Machine Learning and Regression Approach for Predicting the Right Group of Customers for Automobile Industries', justify=LEFT)
title.config(bg='lavender blush', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=200,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=500,y=100)
preprocessButton.config(font=font1) 

analysisButton = Button(main, text="Data Analysis", command=analysis)
analysisButton.place(x=200,y=150)
analysisButton.config(font=font1) 

knnButton = Button(main, text="KNeighborsClassifier", command=custom_knn_classifier)
knnButton.place(x=500,y=150)
knnButton.config(font=font1)

LRButton = Button(main, text="Randomforestclassifier", command=Randomforestclassifier)
LRButton.place(x=200,y=200)
LRButton.config(font=font1)

predictButton = Button(main, text="Prediction", command=Prediction)
predictButton.place(x=200,y=250)
predictButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=500,y=200)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=500,y=250)
exitButton.config(font=font1)

                            

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1) 

main.config(bg='LightSteelBlue1')
main.mainloop()
