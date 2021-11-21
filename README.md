# Dataverse_21_LDAmodel

# Description

This project aims to find the best number of clusters for the data points in the dataset and label each point
to a topic. 

In this unsupervised learning project we aim to preprocess the dataset, prepapre the dataset for LDA moel, optimize model parameters,
and assign possible labels/topics to each unlabeled datapoint.

# Motivation
Reddit comments can get lengthy and often form their own topics. Users end up using search engines such as google to search for relevant topics. Alot of the activity in 
reddit chat rooms about stocks and cryptocrurrency can be used to correlate and detect possible price fluctuations. 

# Dataset

The original/uncleaned dataset is obtained from 2017-11.csv file in https://www.kaggle.com/nickreinerink/reddit-rcryptocurrency?ref=hackernoon.com
The notebook uses a cleaned/preprocessed dataset that we saved in https://www.kaggle.com/samkamarfua/dataverse-21-samka in order to save the run time.
The old_text in the above dataset is orginal text/comment and the clean_text is the preprocessed text.

# How to run it

Due to time constraint alot of the results have been saved in files and then imported in the notebook. In order to run the notbook 'lda-model0.ipynb', we need to download
the preprocessed dataset from https://www.kaggle.com/samkamarfua/dataverse-21-samka. Install packages 'gensim'.  Comment out the code chunk for optimization of coherent values - the parameter values are saved in file optimize_values.csv.
And that should be it!

We tried to keep the notebook simple and minimal with the least amount of installations and run time required!

The 'topic_disp.html' is the interactive site that helps to understand the different topics and how they overlap with each other.


# Credits
This project is completed by Samka Marfua and Vicky Chen for Dataverse 2021. 

# References
https://www.youtube.com/watch?v=IUAHUEy1V0Q&t=654s
https://www.analyticsvidhya.com/blog/2021/06/part-2-topic-modeling-and-latent-dirichlet-allocation-lda-using-gensim-and-sklearn/


