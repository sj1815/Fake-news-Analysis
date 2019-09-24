import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, precision_score
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

def build_LinearSVC_classifier(train_x, train_y, test_x, test_y):
    count_vect = CountVectorizer(stop_words='english', max_df=0.7, analyzer='word', tokenizer=tokenize)
    X_train_counts = count_vect.fit_transform(train_x)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf =  LinearSVC()
    clf = clf.fit(X_train_tfidf, train_y)

    X_test_counts = count_vect.transform(test_x)

    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)
    print("Linear SVC Classifier")
    print("Accuracy of classification: " + str(np.mean(test_y == predicted)))

    print("Confusion matrix:\n" + str(confusion_matrix(test_y, predicted)))
    precision, recall, fscore, Noneval = precision_recall_fscore_support(test_y, predicted, average='macro')
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Fscore: " + str(fscore))

def build_Logistic_Regression_model(train_x, train_y, test_x, test_y):
    count_vect = CountVectorizer(stop_words='english', max_df=0.7, analyzer='word', tokenizer=tokenize)
    X_train_counts = count_vect.fit_transform(train_x)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = LogisticRegression()
    clf = clf.fit(X_train_tfidf, train_y)

    X_test_counts = count_vect.transform(test_x)

    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)
    print("Logistic Regression Classifier")
    print("Accuracy of classification: " + str(np.mean(test_y == predicted)))

    print("Confusion matrix:\n" + str(confusion_matrix(test_y, predicted)))
    precision, recall, fscore, Noneval = precision_recall_fscore_support(test_y, predicted, average='macro')
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Fscore: " + str(fscore))

def build_KNN_classifier(train_x, train_y, test_x, test_y):
    count_vect = CountVectorizer(stop_words='english', max_df=0.7, analyzer='word', tokenizer=tokenize)
    X_train_counts = count_vect.fit_transform(train_x)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf = clf.fit(X_train_tfidf, train_y)

    X_test_counts = count_vect.transform(test_x)

    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)
    print("K-nearest neighbor classifier for k = 1")
    print("Accuracy of classification: " + str(np.mean(test_y == predicted)))

    print("Confusion matrix:\n" + str(confusion_matrix(test_y, predicted)))
    precision, recall, fscore, Noneval = precision_recall_fscore_support(test_y, predicted, average='macro')
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Fscore: " + str(fscore))

def build_Random_forest_classifier(train_x, train_y):
    name = "Random_forest"

    count_vect = CountVectorizer(stop_words='english', max_df=0.7, analyzer='word', tokenizer=tokenize)
    X_train_counts = count_vect.fit_transform(train_x)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    rf = RandomForestClassifier(n_estimators = 10, random_state = 42)
    rf.fit(X_train_tfidf,train_y)
    joblib.dump(rf, name + '_classifier.sav')
    joblib.dump(count_vect, name + '_counts.sav')
    joblib.dump(tfidf_transformer, name + '_tfidf.sav')

def test_Random_forest_classifier(test_x, test_y):
    name = "Random_forest"

    count_vect = joblib.load(name+'_counts.sav')
    X_test_counts = count_vect.transform(test_x)

    tfidf_transformer = joblib.load(name+'_tfidf.sav')
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    rf = joblib.load(name+'_classifier.sav')
    predicted = rf.predict(X_test_tfidf)
    print(name, "classifier")
    print("Accuracy of classification: "+str(np.mean(test_y == predicted)))

    print("Confusion matrix:\n" + str(confusion_matrix(test_y, predicted.astype(np.int))))
    precision, recall, fscore, Noneval = precision_recall_fscore_support(test_y, predicted.astype(np.int), average='macro')
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Fscore: " + str(fscore))

def build_Multinomial_NB_classifier(train_x, train_y):
    name = "single_multi_class"

    count_vect = CountVectorizer(stop_words='english', max_df=0.7, analyzer='word', tokenizer=tokenize)
    X_train_counts = count_vect.fit_transform(train_x)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, train_y)

    joblib.dump(clf, name + '_classifier.sav')
    joblib.dump(count_vect, name + '_counts.sav')
    joblib.dump(tfidf_transformer, name + '_tfidf.sav')

def test_Multinomial_NB_classifier(test_x, test_y):
    name = "single_multi_class"

    count_vect = joblib.load(name+'_counts.sav')
    X_test_counts = count_vect.transform(test_x)

    tfidf_transformer = joblib.load(name+'_tfidf.sav')
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    clf = joblib.load(name+'_classifier.sav')
    predicted = clf.predict(X_test_tfidf)
    print("Naive Bayes Classifier: ")
    print("Accuracy of classification: "+str(np.mean(test_y == predicted)))

    print("Confusion matrix:\n" + str(confusion_matrix(test_y, predicted)))
    precision, recall, fscore, Noneval = precision_recall_fscore_support(test_y, predicted, average='macro')
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Fscore: " + str(fscore))

def stem_lemmatize_tokens(tokens, lemmatizer, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(lemmatizer.lemmatize(stemmer.stem(item)))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stems = stem_lemmatize_tokens(tokens, lemmatizer, stemmer)
    return stems

def see_distribution(df):
    df.info()
    plt.hist(df['label'], 10, histtype='bar', facecolor='blue')
    plt.show()

def main():
    df = pd.read_csv("train.csv", sep=',', encoding='utf-8')

    #data cleaning
    title = df['title']
    title = title.str.lower().replace('[^A-Za-z0-9\s]+', '')
    title = title.fillna('')

    text = df['text']
    text = text.str.lower().replace('[^A-Za-z0-9\s]+', '')
    text = text.fillna('')

    type = df['label'].fillna(0)

    #convert to numeric values
    #le = preprocessing.LabelEncoder()
    #le.fit(['bs', 'bias', 'conspiracy', 'hate', 'junksci', 'satire', 'state', 'fake'])
    #le.fit(['reliable', 'unreliable'])
    #type = le.transform(type)
    #type = type.reshape(-1,1)

    #see_distribution(df)
    train_x, test_x, train_y, test_y = train_test_split(title, type, train_size=0.7)


    """
    Naive Bayes
    """
    build_Multinomial_NB_classifier(train_x, train_y)
    test_Multinomial_NB_classifier(test_x, test_y)
    print()
    """
    Random Forests
    """
    build_Random_forest_classifier(train_x, train_y)
    test_Random_forest_classifier(test_x, test_y)
    print()
    """
    KNN k = 1
    """
    build_KNN_classifier(train_x, train_y, test_x, test_y)
    print()
    """
    Logistic Regression
    """
    build_Logistic_Regression_model(train_x, train_y, test_x, test_y)
    print()
    """
    Linear SVC
    """
    build_LinearSVC_classifier(train_x, train_y, test_x, test_y)
    print()



if __name__ == '__main__':
    main()

