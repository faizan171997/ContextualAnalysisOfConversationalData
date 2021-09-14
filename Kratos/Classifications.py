import pickle
import string

import pandas as pd


def training_models(filename):
    """
    Used for the classification of the sentences
    :parameter training_models (filename)
    :return classification
    """

    print("training data")
    news_df = pd.read_csv(filename, sep=",")
    news_df['CATEGORY'] = news_df.CATEGORY.map({'b': 1, 't': 2, 'e': 3, 'm': 4, 'p': 5, 's': 6})
    news_df['TITLE'] = news_df.TITLE.map(
        lambda x: x.lower().translate(str.maketrans('', '', string.punctuation))
    )
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        news_df['TITLE'],
        news_df['CATEGORY'],
        random_state=1
    )
    from sklearn.feature_extraction.text import CountVectorizer

    count_vector = CountVectorizer(stop_words='english')
    training_data = count_vector.fit_transform(X_train)
    # ---Storing Vectorization---------
    pickle.dump(count_vector, open("Kratos/vectorizer.pkl", "wb"))


    testing_data = count_vector.transform(X_test)

    from Kratos import Classifications

    result = Classifications.naive_bayes(training_data, y_train, testing_data,y_test)
    return result

def model_analysis(filename,testing_data,y_test):
    """
    :param filename: pickle stored model
    :param testing_data: (X_test) testing data
    :param y_test: label of text
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    predictions_nb = loaded_model.predict(testing_data)

    from sklearn.metrics import confusion_matrix

    cnf_matrix = confusion_matrix(y_test, predictions_nb)
    print(cnf_matrix)

    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    print("Model Analysis")
    print("Accuracy score: ", accuracy_score(y_test, predictions_nb))
    print("Recall score: ", recall_score(y_test, predictions_nb, average='weighted'))
    print("Precision score: ", precision_score(y_test, predictions_nb, average='weighted'))
    print("F1 score: ", f1_score(y_test, predictions_nb, average='weighted'))



def naive_bayes(training_data, y_train, input_data,y_test):
    """

    :param training_data:
    :param y_train:
    :param input_data:
    :param y_test:
    :return:
    """
    from sklearn.naive_bayes import MultinomialNB
    print("NaiveBayes")

    filename = "Kratos/stored_model/naive_bayes.pkl"
    naive_bayes = MultinomialNB()
    naive_bayes.fit(training_data, y_train)
    pickle.dump(naive_bayes, open(filename, 'wb'))
    model_analysis(filename,input_data,y_test)

def svm(training_data, y_train, input_data,y_test):
    """

    :param training_data:
    :param y_train:
    :param input_data:
    :param y_test:
    :return:
    """
    print("SVM")
    from sklearn.svm import SVC
    filename = "Kratos/stored_model/SVM.pkl"
    print("trainning SVM")

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(training_data, y_train)
    pickle.dump(naive_bayes, open(filename, 'wb'))
    model_analysis(filename, input_data, y_test)


def random_forest(training_data, y_train, input_data,y_test):
    """
    random forest training
    :param training_data:
    :param y_train:
    :param input_data:
    :param y_test:
    :return: model analisys
    """
    print("rf")
    from sklearn.ensemble import RandomForestRegressor
    filename = "Kratos/stored_model/random_forest.pkl"

    regressor = RandomForestRegressor(n_estimators=2, random_state=0)
    regressor.fit(training_data, y_train)
    pickle.dump(naive_bayes, open(filename, 'wb'))
    model_analysis(filename, input_data, y_test)

def decision_tree(training_data, y_train, input_data,y_test):
    """
    Decision tree
    :param training_data:
    :param y_train:
    :param input_data:
    :param y_test:
    :return:
    """
    print("decision tree")
    from sklearn.tree import tree
    filename = "Kratos/stored_model/decision_tree.pkl"
    print("Training Decison tree")
    regressor = tree.DecisionTreeClassifier()
    regressor.fit(training_data, y_train)
    pickle.dump(regressor, open(filename, 'wb'))

    print("Trained Decison tree")
    model_analysis(filename, input_data, y_test)

def get_context(filename,testing_data):
    """
    it will give context
    :param filename: stored model of algorithm
    :param testing_data: X_test
    :return: predictions (Y)
    """

    count_vector=pickle.load(open("Kratos/vectorizer.pkl","rb"))
    print("loaded")
    testing_data = count_vector.transform(testing_data)
    loaded_model = pickle.load(open(filename, 'rb'))
    predictions = loaded_model.predict(testing_data)
    return predictions
    # return predictions
