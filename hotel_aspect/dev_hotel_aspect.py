# -*- coding: utf-8 -*-
import joblib
import numpy as np
import re
import os
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import  LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# PATH = os.path.abspath(os.path.join(__file__, "./../"))
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


class HotelAspect:
    def __init__(self, dirpath=PATH):
        """
        Initialize the HotelAspect class.

        Parameters:
        - dirpath (str): Path to the directory containing necessary files.
        """
        self.dirpath = dirpath
        self.models = None
        self.vectorizers = None
        self.regex = re.compile('<.*?>')
        self.aspect_keywords = self.load_aspect_keywords()

        self.load_dict(os.path.join(dirpath, 'data','dict'))

        self.aspects = ["surrounding", "service", "meal", "location",
        "staff", "room", "facility", "quality", "value"]
        self.stemmer = PorterStemmer()
        self.load_models(dirpath)

    def get_classes(self):
        """
        Get classes from the model.

        Returns:
        - classes (list): List of classes from the model.
        """

        return self.model.classes_
    
    def load_aspect_keywords(self,data = 'data/hotel-aspect-keywords.xlsx',sheet_name = 'english_positive'):
        """
        Load aspect keywords for helping inference.

        Parameters:
        - data (str): Path to the Excel file containing aspect keywords.

        Returns:
        - aspect_keywords (dict): Dictionary of aspect keywords.
        """

        aspect_keywords = pd.read_excel(data,sheet_name=sheet_name).to_dict(orient='list')
        aspect_keywords = {key: [value for value in values if pd.notna(value)] for key, values in aspect_keywords.items()}
        return aspect_keywords


    def load_models(self, dirpath = PATH, save_path = "models_english"):
        """
        Load models and vectorizers for each aspect.

        Parameters:
        - dirpath (str): Path to the directory containing models and vectorizers.
        """

        self.models = {}
        self.vectorizers = {}

        # Iterate through each aspect and load the corresponding model and vectorizer
        for aspect in self.aspects:
            # Construct file paths for vectorizer and classifier
            vect = f"tfidf-{aspect}.joblib"
            clf = f"clf-{aspect}.joblib"

            # Load the TF-IDF vectorizer
            model_tfidf = joblib.load(os.path.join(dirpath,save_path, vect))

            # Load the classifier model
            model_clf = joblib.load(os.path.join(dirpath, save_path, clf))

            # Store the loaded models and vectorizers in the class attributes
            self.models[aspect] = model_clf
            self.vectorizers[aspect] = model_tfidf

    def train_test(self,texts,sentiments, with_score=True,class_weight='balanced'):
        """
        Train and test the model.

        Parameters:
        - texts (pd.Series): Text data for training.
        - sentiments (pd.DataFrame): Sentiments corresponding to each aspect.
        - with_score (bool) : If true, the model will use score_aspect for inferencing
        - class_weight : dict or 'balanced', or None, default='balanced'
                         Weights associated with classes in the form {class_label: weight}. 
                         If not given, all classes are supposed to have weight balanced
        """

        self.aspects = sentiments.columns
        texts = texts.apply(self.clean_text)
               
        # Iterate through all aspects for training and testing per aspect
        for aspect in self.aspects:
            print(f'{aspect} aspect')
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(texts, sentiments[aspect], test_size=0.1, stratify=sentiments[aspect], random_state=42)
            
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer()
            X_train_vect = vectorizer.fit_transform(X_train)
            
            # Store the vectorizer for later use
            self.vectorizers[aspect] = vectorizer
            
            
            # Initialize classifiers with balanced class weights and set a random seed for reproducibility
            svm = LinearSVC(class_weight=class_weight, dual='auto', random_state=42)
            rf = RandomForestClassifier(class_weight=class_weight, random_state=42)
            lr = LogisticRegression(multi_class='multinomial', class_weight=class_weight, random_state=42)
            gb = GradientBoostingClassifier(random_state=42)

            # Combine all models using Voting Classifier
            ensemble = VotingClassifier(estimators=[('lr', lr), ('svm', svm), ('rf', rf), ('gb', gb)], voting='hard', n_jobs=-1)

            # Fit the ensemble model to the training data
            ensemble.fit(X_train_vect, y_train)

            # Store the trained model in the class attribute
            self.models[aspect] = ensemble

            # Make predictions on the testing set and evaluate the model
            y_pred = []
            #if using score aspect for inference 
            if with_score:
                for text in X_test:
                    score_aspect = self.score_text_on_aspect(text, self.aspect_keywords)
                    y_pred.append(self.predict_aspect_with_score(text, aspect, score_aspect))
            
            #if not using score aspect for inference 
            else :
                for text in X_test:
                    y_pred.append(self.predict_aspect(text, aspect))

            # Evaluate the model on the testing set
            self.evaluate_model(y_test, y_pred,aspect)

    def create_models(self, texts, sentiments,with_score=True,save_path = "models_english"):
        """
        Train models and save them.

        Parameters:
        - texts (pd.Series): Text data for training.
        - sentiments (pd.DataFrame): Sentiments corresponding to each aspect.
        """

        # Train and test models
        self.train_test(texts, sentiments, with_score)

        # Save the trained models
        self.save_model(save_path=save_path)

    def save_model_aspect(self, vectorizer, model, aspect,save_path = "models_english"):
        """
        Save the trained vectorizer and model for a specific aspect.

        Parameters:
        - vectorizer: Trained TF-IDF vectorizer.
        - model: Trained machine learning model.
        - aspect (str): Aspect for which the model is trained.

        Example:
        - save_model_aspect(vectorizer, model, 'location')
        - Saved Files: 'tfidf-location', 'clf-location'
        """

        print(f'Saving model {aspect}')
        joblib.dump(vectorizer, os.path.join(self.dirpath, f'{save_path}', "tfidf-%s.joblib" % aspect))
        joblib.dump(model, os.path.join(self.dirpath, f'{save_path}', "clf-%s.joblib" % aspect))


    def save_model(self,save_path = "models_english"):
        """
        Save all trained vectorizers and models for each aspect.

        Example:
        - save_model()
        - Saved Files: 'tfidf-facility', 'clf-facility', 'tfidf-location', 'clf-location', ...
        """

        for aspect in self.aspects:
            self.save_model_aspect(self.vectorizers[aspect], self.models[aspect], aspect,save_path=save_path)
        print('Save model Finished')



    def evaluate_model(self,y_test,y_pred,aspect):
        """
        Evaluate the model performance.

        Parameters:
        - y_test (list): True labels.
        - y_pred (list): Predicted labels.
        """
        
        #Lower every row in both label data
        y_test = y_test.apply(lambda row: row.lower())

        cm_labels = np.unique(y_test)

        # Calculate confusion matrix
        cm_array = confusion_matrix(y_test, y_pred)
        cm_array_df = pd.DataFrame(cm_array, index=cm_labels, columns=cm_labels)

        # Print accuracy, classification report, and visualize the confusion matrix
        print("Accuracy\n", accuracy_score(y_test, y_pred) * 100, "%")
        print("\nclassification_report\n", classification_report(y_test, y_pred))

        # Visualize the confusion matrix using a heatmap
        sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, cmap="crest", fmt='d')
        plt.title(f'Confusion matrix for {aspect} aspect')
        plt.xlabel(f'Prediction')
        plt.ylabel(f'True')
        plt.show()


    def load_dict(self, dict_path):
        """
        Load English stopwords from a specified file.

        Parameters:
        - dict_path (str): The path to the directory containing the stopwords file.

        """

        with open(os.path.join(dict_path, 'en_stopwords.txt'), 'r') as f:
            en_stopwords = f.read().splitlines()
        
        # Store the loaded stopwords in the class attribute
        self.stopwords = en_stopwords

    def clean_html(self, raw_text):
        """
        Clean HTML tags from the raw text.

        Parameters:
        - raw_text (str): Raw text containing HTML tags.

        Returns:
        - cleantext (str): Text without HTML tags.
        """

        cleantext = re.sub(self.regex, '', raw_text)
        return cleantext

    def clean_text(self, text):
        """
        Clean and preprocess the input text.

        Parameters:
        - text (str): Input text.

        Returns:
        - temp (str): Processed text.
        """

        temp = str(text).lower()
        temp = re.sub('\n', " ", temp)
        temp = re.sub('\'', "", temp)
        temp = re.sub('-', " ", temp)
        temp = re.sub("[^a-z]", " ", str(temp))
        temp = temp.split()
        temp = [w for w in temp if not w in self.stopwords]
        temp = " ".join(word for word in temp)
        temp = " ".join([self.stemmer.stem(word) for word in temp.split()])

        return temp

    def preprocessing(self, name, desc):
        """
        Preprocess the input name and description.

        Parameters:
        - name (str): Hotel name.
        - desc (str): Hotel description.

        Returns:
        - text (str): Processed text.
        """

        text = self.clean_text(name) + " "+self.clean_text(desc)
        return text

    def predict_aspect_with_score(self, text, aspect, score):
        """
        Predict the sentiment for a given aspect if score given.

        Parameters:
        - text (str): Input text to predict sentiment for.
        - aspect (str): The aspect for which sentiment is predicted.
        - score (dict): Aspect scores based on keywords.

        Returns:
        - prediction (str or np.nan): Predicted sentiment ('Positive', 'Negative', or np.nan if not applicable).
        """
        # Clean the input text
        text = self.clean_text(text)

        # Check if the aspect score is positive, negative, or neutral
        if score[aspect] > 0:
            return 'positive'
        elif score[aspect] < 0:
            return 'negative'
        else:
            # If score = 0, use the trained model to make a prediction
           pred = self.predict_aspect(text,aspect)
           return pred
        
    def predict_aspect(self,text,aspect):
        """
        Predict the sentiment for a given aspect if score is not given.

        Parameters:
        - text (str): Input text to predict sentiment for.
        - aspect (str): The aspect for which sentiment is predicted.

        Returns:
        - prediction (str or np.nan): Predicted sentiment ('Positive', 'Negative', or np.nan if not applicable).
        """

        vectorized = self.vectorizers[aspect].transform([text])
        pred = self.models[aspect].predict(vectorized)

        # Check for special case where prediction is "na"
        if pred[0] == "na":
            # if the pred is Nan, then the sentiment is neutral
            return 'neutral'
        else:
            return pred[0].lower()


    def predict(self, name, desc, with_score=True):
        """
        Predict sentiments for all aspects.

        Parameters:
        - name (str): Hotel name.
        - desc (str): Hotel description.

        Returns:
        - result (dict): Predicted sentiments for each aspect.
        """

        #text = self.preprocessing(name, desc)
        score_aspect = self.score_text_on_aspect(name+""+desc,self.aspect_keywords)

        result = {}

        # Iterate through predefined aspects
        if with_score:
            for aspect in self.aspects:
            # Predict sentiment for each aspect
                result[aspect] = self.predict_aspect_with_score(name + ' ' + desc, aspect, score_aspect)
        else:
            for aspect in self.aspects:
            # Predict sentiment for each aspect
                result[aspect] = self.predict_aspect(name + ' ' + desc, aspect)
        return result
    
    def score_text_on_aspect(self,text, aspect_keywords):
        """
        Score the input text based on aspect keywords.

        Parameters:
        - text (str): Input text.
        - aspect_keywords (dict): Aspect keywords.

        Returns:
        - aspect_value (dict): Aspect scores.

        Example:
        - Input: score_text_on_aspect("Great location and friendly staff", aspect_keywords)
        - Output: {'location': 1, 'staff': 1}
        """
    
        # Convert text to lowercase for case-insensitive matching
        lower_text = text.lower()

        # Initialize dictionaries to store aspect sentiment and scores
        aspect_sentiment = {}
        aspect_value = {}

        # Iterate through predefined aspects
        for aspect, keywords in aspect_keywords.items():
            aspect_sentiment[f'{aspect}'] = 0

            # Count occurrences of keywords in the text
            for keyword in keywords:
                if keyword.lower() in lower_text:
                    aspect_sentiment[f'{aspect}'] += 1
                else:
                    pass

            # Store score per aspect
            if aspect.split('_')[1] == 'negative':
                aspect_name = aspect.split('_')[0]
                aspect_value[f'{aspect_name}'] = aspect_sentiment[f'{aspect_name}_positive'] - aspect_sentiment[f'{aspect_name}_negative']

        return aspect_value