# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <div style="background-color: lightblue; padding: 40px; font-size: 40px;">
#              Classification Model Evaluation
# </div>

#    Classification is a technique for labeling the class of an observation.Classification can be used to predict        binary classes (2 classes) or multi-classes (>2 classes).
#
#    Classification Model: A series of steps that takes the patterns of input variables, generalizes those patterns,    and applies them to new data in order to predict the class.
#    
#    A confusion matrix is a cross-tabulation of our model's predictions against the actual values. 
#   
#    Accuracy evaluates how many correct predictions (both positive and negative) were made over the total number of    predictions.
#   
#    Precision evaluates how many of the positive predictions were correct.
#   
#    Recall evaluates how the model handled all positive outcomes.
#   
#    Misclassification rate concerns how many predictions were incorrect. This accounts for all other outcomes not      included in the calculation of accuracy.
#   
#    Sensitivity (True Positive Rate) TP/(TP+FN)
#
#    How well does the model predict negative outcomes? Specificity TN/(FP+TN)
#
#    Negative Predictive Value  TN/(FN+TN)
#   
#    F1 Score  2*((Precision*Recall)/(Precision+Recall))
#
#    The baseline is a simple model that is a reference point for the performance of other models.
#    For a classification model, a baseline is often the mode.
#

Designation	Description
True Negative	Model correctly predicted the negative outcome
False Positive	Model incorrectly predicted the positive outcome
False Negative	Model incorrectly predicted the negative outcome
True Positive	Model correctly predicted the positive outcome

# ### Exercises
#

# ### 1. Create a new file named model_evaluation.py or model_evaluation.ipynb for these exercises.

# ### 2.Given the following confusion matrix, evaluate (by hand) the model's performance.
#
#
# |               | pred dog   | pred cat   |
# |:------------  |-----------:|-----------:|
# | actual dog    |         46 |         7  |
# | actual cat    |         13 |         34 |
# In the context of this problem, what is a false positive?
# In the context of this problem, what is a false negative?
# How would you describe this model?

# Dog is negative class
# Cat is positive class

# ### 2.1 In the context of this problem, what is a false positive? 
#
# False Positive: We predicted a cat but it is a dog.
#
# The predicted false positive is 7.   

# ### 2.2 In the context of this problem, what is a false negative?
#
# False Negative: We predicted a dog but it is a cat.
#
# The predicted false negative is 13.  

# ### 2.3 How would you describe this model?
#
# Since we are setting Cat is positive class and Dog is negative class:
#
#
# True positive would be predicting a cat and it is an actual cat
#
# True negative would be predicting a dog and it is an actual dog
#
# False positive would be predicting a cat and it is a dog
#
# False negative would be predicting a dog and it is a cat


# +
#true positive is predicting its a cat, and its a cat
tp = 34

#true negative is predicting its a dog, and its a dog
tn = 46

#false positive is predicting its a cat, but its a dog
fp = 7

#false negative is predicting its a dog, but its a cat
fn = 13

# +

print("Cat-classifier (i.e 'cat' is the positive prediction)")

print("False Negatives:", fn)
print("True Negatives:", tn)
print("True Positives:", tp)
print("False Positives:", fp)
print("===========================")

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Accuracy is:", accuracy)
print("Recall is:", round(recall,2))
print("Precision is:", round(precision,2))      
# -

# ### 3. You are working as a datascientist working for Codeup Cody Creator (C3 for short), a rubber-duck manufacturing plant.
#
# Unfortunately, some of the rubber ducks that are produced will have defects. Your team has built several models that try to predict those defects, and the data from their predictions can be found here.
#
# Use the predictions dataset and pandas to help answer the following questions:
#
# An internal team wants to investigate the cause of the manufacturing defects. They tell you that they want to identify as many of the ducks that have a defect as possible. Which evaluation metric would be appropriate here? Which model would be the best fit for this use case?
#
#
# Recently several stories in the local news have come out highlighting customers who received a rubber duck with a defect, and portraying C3 in a bad light. The PR team has decided to launch a program that gives customers with a defective duck a vacation to Hawaii. They need you to predict which ducks will have defects, but tell you the really don't want to accidentally give out a vacation package when the duck really doesn't have a defect. Which evaluation metric would be appropriate here? Which model would be the best fit for this use case?
#

import pandas as pd

# +
# Download the c3.csv into your current folder
#load dataframe
ccc_df = pd.read_csv('c3.csv')

#take a look
ccc_df.head()
# -

# ### 3. a. An internal team wants to investigate the cause of the manufacturing defects. They tell you that they want to identify as many of the ducks that have a defect as possible. Which evaluation metric would be appropriate here? Which model would be the best fit for this use case?

#checking the no: of defects and non-defects in the actual data
ccc_df.actual.value_counts()

# comparing the distribution of categorical variables in the dataset for model1
pd.crosstab(ccc_df.actual, ccc_df.model1)


# comparing the distribution of categorical variables in the dataset for model2
pd.crosstab(ccc_df.actual, ccc_df.model2)


# comparing the distribution of categorical variables in the dataset for model3
pd.crosstab(ccc_df.actual, ccc_df.model3)


# +
As we are interested in defect we will assign it as positive class for the classifier.

defects = positive class
Our best metric here is recall = tp/(tp + fn)

how many real positives do we have?
how many of defective ducks are actually flagged by defective (positive) by the models?
let's minimize our false negatives
# -

# Model positives
subset = ccc_df [ccc_df.actual == 'Defect']
subset.head()

# +
# model1
model1_recall = (subset['actual'] == subset['model1']).mean()

# model2
model2_recall = (subset['actual'] == subset['model2']).mean()
#model3
model3_recall = (subset['actual'] == subset['model3']).mean()

print(f"Model 1 Recall: {model1_recall:.2%} \n Model 2 Recall: {model2_recall:.2%} \n Model 3 Recall: {model3_recall:.2%} \n")

'''
Choose Recall as the metric and Model 3 as the optimal model

'''

# +
# OR
# -

#Model 1 recall
(subset.actual == subset.model1).mean()

#Model 2 recall
(subset.actual == subset.model2).mean()

#Model 3 recall
(subset.actual == subset.model3).mean()

Quality Control should select a model with higher recall (to avoid false negatives)
Quality Control should use Model 3

# ### b. Recently several stories in the local news have come out highlighting customers who received a rubber duck with a defect, and portraying C3 in a bad light. The PR team has decided to launch a program that gives customers with a defective duck a vacation to Hawaii. They need you to predict which ducks will have defects, but tell you the really don't want to accidentally give out a vacation package when the duck really doesn't have a defect. Which evaluation metric would be appropriate here? Which model would be the best fit for this use case?

# +
# pull positive predictions from each model
# subset the data for positives from each model

#model1 precision
subset = cody_df[cody_df.model1=='Defect']
model1_precision = (subset.actual == subset.model1).mean()

# choose subset for model2 where we only select 'positive predictions'
subset2 = cody_df [cody_df.model2 == 'Defect']
# calculate precision
model2_precision = (subset2.actual == subset2.model2).mean()

# choose subset for model3 where we only select 'positive predictions'
subset3 = cody_df [cody_df.model3 == 'Defect']
# calculate precision
model3_precision = (subset3.actual == subset3.model3).mean()

print(f"""Model 1 Precision: {model1_precision:.2%}
Model 2 Precision: {model2_precision:.2%}
Model 3 Precision: {model3_precision:.2%}
""")

# +
# OR

# +
PR team really wants to minimize the False positives means choose the model with the highest precision.

So the best models for this scenario is precision = tp / (tp + fp)

defect = positive class

# +
# choose subset for model 1 where we only select 'positive predictions'
subset = ccc_df [ccc_df.model1 == 'Defect']

# calculate precision
(subset.actual == subset.model1).mean()

# +
#Choose subset for model2 where we only select 'positive predictions'
subset = ccc_df[ccc_df.model2 == 'Defect']

#calculate precision
(subset.actual == subset.model2).mean()

# +
#Choose subset for model3 where we only select 'positive predictions'
subset = ccc_df[ccc_df.model3 == 'Defect']

#calculate precision
(subset.actual == subset.model3).mean()
# -

Use model 1 as it has the highest precision, hence it will minimize the false positive predictions of defects

# ### 4.You are working as a data scientist for Gives You Paws ™, a subscription based service that shows you cute pictures of dogs or cats (or both for an additional fee).At Gives You Paws, anyone can upload pictures of their cats or dogs. The photos are then put through a two step process. First an automated algorithm tags pictures as either a cat or a dog (Phase I). Next, the photos that have been initially identified are put through another round of review, possibly with some human oversight, before being presented to the users (Phase II). Several models have already been developed with the data, and you can find their results here. Given this dataset, use pandas to create a baseline model (i.e. a model that just predicts the most common class) and answer the following questions:
#

# +
#Download the gives_you_paws.csv into your current folder
# load dataframe
paws_df = pd.read_csv('gives_you_paws.csv')

# take a look
paws_df.head()
# -

#Look at the kind of columns and dtypes we are dealing with
paws_df.info()

# our actual counts
paws_df.actual.value_counts()

#set the most common class ('dog') as the baseline
paws_df['baseline'] = paws_df.actual.value_counts().idxmax()
paws_df.head()

# ### 4. a. In terms of accuracy, how do the various models compare to the baseline model? Are any of the models better than the baseline?

#
# cat = positive class
# dog = negative class

# +
print("Baseline",( paws_df['actual'] == paws_df['baseline']).mean())

print("Model1",( paws_df['actual'] == paws_df['model1']).mean())

print("Model2",( paws_df['actual'] == paws_df['model2']).mean())

print("Model3",( paws_df['actual'] == paws_df['model3']).mean())

print("Model4",( paws_df['actual'] == paws_df['model4']).mean())


# +
# OR
# -

#baseline accuracy 
(paws_df.actual == paws_df.baseline).mean()

#model 1 accuracy
(paws_df.model1 == paws_df.actual).mean()


#model 2 accuracy
(paws_df.model2 == paws_df.actual).mean()


#model 3 accuracy
(paws_df.model3 == paws_df.actual).mean()


#model 4 accuracy
(paws_df.model4 == paws_df.actual).mean()


In terms of accuracy, model 1 and model 4 perform better than baseline

# ### 4. b. Suppose you are working on a team that solely deals with dog pictures. Which of these models would you recommend?
#
# dog = positive class
# cat = negative class
# Two-phases - recall = tp/(tp + fn) and precision = tp/(tp + fp)

# +
# subsetting the data for just dogs
# we are using reality, so use recall

# +
# For Phase I, choose a model with highest recall

subset = paws_df[paws_df.actual == 'dog']
subset.head()

# +
model_recall = []

# for everything except actual so from model1 and on... do something
for model in subset.columns[1:]:
    recall = (subset.actual == subset[model]).mean()
    model_recall.append([model, recall])
    
model_recall

# +
# OR
# -

# Model 1 Recall
(subset.actual == subset.model1).mean()

# Model 2 Recall
(subset.actual == subset.model2).mean()

# Model 3 Recall
(subset.actual == subset.model3).mean()

# Model 4 Recall
(subset.actual == subset.model4).mean()

Model 4 is performing the best, with Recall of 0.96

# ### For Phase II, choose a model with highest precision

subset1 = paws_df[paws_df.model1 == 'dog']
subset2 = paws_df[paws_df.model2 == 'dog']
subset3 = paws_df[paws_df.model3 == 'dog']
subset4 = paws_df[paws_df.model4 == 'dog']

print("Subset 1: ", (subset1.actual == subset1.model1).mean())
print("Subset 2: ", (subset2.actual == subset2.model2).mean())
print("Subset 3: ", (subset3.actual == subset3.model3).mean())
print("Subset 4: ", (subset4.actual == subset4.model4).mean())

Model 2 performs the best in terms of our evaluation metric: Precision of 89.3%

# +
# OR

# +
# For Phase II, choose a model with highest precision

subset = paws_df[paws_df.actual == 'dog']
subset.head()
# -

#take another look
paws_df.head()

# +
#subset +ive classes in each model to calculate precision
# -

subset1 = paws_df[paws_df.model1 == 'dog']
subset2 = paws_df[paws_df.model2 == 'dog']
subset3 = paws_df[paws_df.model3 == 'dog']
subset4 = paws_df[paws_df.model4 == 'dog']

# Model 1 Precision
(subset1.actual == subset1.model1).mean()


# Model 2 Precision
(subset2.actual == subset2.model2).mean()

# Model 3 Precision
(subset3.actual == subset3.model3).mean()

# Model 4 Precision
(subset4.actual == subset4.model4).mean()

Model 2 and Model 1 are performing best with Precision of 0.893

# ### 4. c. Suppose you are working on a team that solely deals with cat pictures. Which of these models would you recommend?
#
# dog = negative class
# cat = postive class
# Two-phases - recall = tp/(tp + fn) and precision = tp/(tp + fp)

#paws_df.drop(columns=['baseline'], inplace=True)
paws_df['baseline'] = 'cat'
paws_df.head()

# +
#phaseII: precision

model_prec = []


for model in paws_df.columns[1:]:
    subset = paws_df[paws_df[model] == 'cat']
    
    precision = (subset.actual == subset[model]).mean()
    
    
    model_prec.append([model, precision])
    
model_prec
# -

Model 4 is best in terms of recall

# +
# Phase I recall

subset = paws_df[paws_df.actual=='cat']

model_recall = []


for model in subset.columns[1:]:
    subset = paws_df[paws_df[model] == 'cat']
    
    recall = (subset.actual == subset.model1).mean()

    
    model_recall.append([model, recall])
    
model_recall

# -

Model 4 is best in terms of Precision

# +
# OR

# +
# For Phase I, choose a model with highest recall

subset = paws_df[paws_df.actual == 'cat']
subset.head()
# -

# Model 1 Recall
(subset.actual == subset.model1).mean()

# Model 2 Recall
(subset.actual == subset.model2).mean()

# Model 3 Recall
(subset.actual == subset.model3).mean()

# Model 4 Recall
(subset.actual == subset.model4).mean()

Model 2 is performing the best, with Recall of 0.89

subset1 = paws_df[paws_df.model1 == 'cat']
subset2 = paws_df[paws_df.model2 == 'cat']
subset3 = paws_df[paws_df.model3 == 'cat']
subset4 = paws_df[paws_df.model4 == 'cat']

# Model 1 Precision
(subset1.actual == subset1.model1).mean()

# Model 2 Precision
(subset2.actual == subset2.model2).mean()

# Model 3 Precision
(subset3.actual == subset3.model3).mean()

# Model 4 Precision
(subset4.actual == subset4.model4).mean()

Model 4 is performing the best, with precision of .81

# ### 5. Follow the links below to read the documentation about each function, then apply those functions to the data from the previous problem.
#
# sklearn.metrics.accuracy_score
#
# sklearn.metrics.precision_score
#
# sklearn.metrics.recall_score
#
# sklearn.metrics.classification_report

from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

# +
print("Model 1")

pd.DataFrame(classification_report(paws_df.actual, paws_df.model1,
                                  labels =['cat','dog'],
                                  output_dict=True)).T

# +
print("Model 2")

pd.DataFrame(classification_report(paws_df.actual, paws_df.model2,
                                  labels =['cat','dog'],
                                  output_dict=True)).T

# +
print("Model 3")

pd.DataFrame(classification_report(paws_df.actual, paws_df.model3,
                                  labels =['cat','dog'],
                                  output_dict=True)).T

# +
print("Model 4")

pd.DataFrame(classification_report(paws_df.actual, paws_df.model4,
                                  labels =['cat','dog'],
                                  output_dict=True)).T
# -

from sklearn.metrics import precision_score, recall_score, accuracy_score

paws_df.head()

precision_score(paws_df['actual'], paws_df['model1'], pos_label='dog')

recall_score(paws_df['actual'], paws_df['model1'], pos_label='dog')

accuracy_score(paws_df['actual'], paws_df['model1'])


# +
# OR
# -

def calculate_precision(predictions, positive='dog'):
    return precision_score(paws_df.actual, predictions, pos_label=positive)


def calculate_recall(predictions, positive='dog'):
    return precision_score(paws_df.actual, predictions, pos_label=positive)


def calculate_accuracy(predictions, positive='dog'):
    return precision_score(paws_df.actual, predictions, pos_label=positive)


def calculate_classification(predictions, positive='dog'):
    return precision_score(paws_df.actual, predictions, pos_label=positive)


pd.concat([
   paws_df.loc[:, 'model1':'baseline'].apply(calculate_recall).rename('recall'),
   paws_df.loc[:, 'model1':'baseline'].apply(calculate_precision).rename('precision'),
   paws_df.loc[:, 'model1':'baseline'].apply(calculate_accuracy).rename('accuracy'),
   paws_df.loc[:, 'model1':'baseline'].apply(calculate_accuracy).rename('classification'), 
], axis=1)

# +

# precision_score?

# -

from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report


def calculate_precision(predictions, positive='cat'):
    return precision_score(paws_df.actual, predictions, pos_label=positive)


def calculate_recall(predictions, positive='cat'):
    return precision_score(paws_df.actual, predictions, pos_label=positive)


def calculate_accuracy(predictions, positive='cat'):
    return precision_score(paws_df.actual, predictions, pos_label=positive)


def calculate_classification(predictions, positive='cat'):
    return precision_score(paws_df.actual, predictions, pos_label=positive)


pd.concat([
   paws_df.loc[:, 'model1':'baseline'].apply(calculate_recall).rename('recall'),
   paws_df.loc[:, 'model1':'baseline'].apply(calculate_precision).rename('precision'),
   paws_df.loc[:, 'model1':'baseline'].apply(calculate_accuracy).rename('accuracy'),
   paws_df.loc[:, 'model1':'baseline'].apply(calculate_accuracy).rename('classification'), 
], axis=1)


