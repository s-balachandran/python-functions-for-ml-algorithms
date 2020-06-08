# -*- coding: utf-8 -*-
"""
Evaluation Function for Binary Classification models

Returns confusion matrix, classification accuracy, null accuracy,
precision, recall, specificity, and F score.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from sklearn.metrics import accuracy_score


def evaluation(test_y, predictions):
    
    """
    Takes the test target variable, and the predictions of
    the target variable.
    
    Returns  confusion matrix, classification accuracy, null accuracy,
    precision, recall, specificity, and F score.
    """
    
    #accuracy score
    accuracy = accuracy_score(test_y, predictions)
    print("The classification accuracy is {:.2f} %." .format(accuracy*100))
    
  
    y_test_mean = test_y.mean()
    #null accuracy
    null_accuracy = max(y_test_mean, 1-y_test_mean)
    print('The null accuracy is {:.2f} %.'.format(null_accuracy*100))
    
    #confusion matrix
    skplt.metrics.plot_confusion_matrix(test_y, predictions)
    bottom, top = plt.ylim() 
    bottom += 0.5 
    top -= 0.5 
    plt.ylim(bottom, top)
    plt.yticks(rotation = 45)
    plt.show()
    
    conf_matrix = confusion_matrix(test_y, predictions)
    
    TN = conf_matrix[0,0] #true negatives
    FP = conf_matrix[0,1] #false positives
    FN = conf_matrix[1,0] #false negatives
    TP = conf_matrix[1,1] #true positives
    
    #precision
    precision = TP/(TP+FP)*100
    print('The precision is {:.2f} %.'.format(precision))
    #sensitivity/ recall
    recall = TP/(FN+TP)*100
    print('The sensitivity/recall is {:.2f} %.'.format(recall))
    #specificity
    specificity = TN/(FP+TN)*100
    print('The specificity is {:.2f} %.'.format(specificity))
    #F_score
    F_score = (2*precision*recall)/(precision + recall)
    print('The F score is {:.2f} %.'.format(F_score))
    
    return None