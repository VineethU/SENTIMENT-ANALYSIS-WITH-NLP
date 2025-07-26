# SENTIMENT-ANALYSIS-ON-DATASET-OF-CUSTOMER-REVIEWS

COMPANY: CODTECH IT SOLUTIONS

NAME: UPPU VINEETH

INTERN ID: CTO6DZ1367

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH


Project Overview (Simplified)
This project is about understanding whether a review is positive or negative using machine learning.

I created a small dataset of 40 reviews – 20 positive and 20 negative.

Each review is labeled as 1 (positive) or 0 (negative).

These reviews are short, like real feedback you'd find online.

The goal is to teach a machine how to guess the sentiment of a new review.

Why I did this
Reading and analyzing lots of reviews manually takes too much time.

So, I used sentiment analysis to do it automatically.

This is useful for businesses to understand customer opinions quickly.

How I did it
First, I split the reviews into training and test sets (70-30 split).

Then, I used TF-IDF Vectorizer to convert text to numbers.

After that, I trained a Logistic Regression model using the training data.

I tested the model on the test set and checked how well it did.

I used accuracy, precision, recall, and F1-score for evaluation.

Tools I used:


#.Python


Pandas


Scikit-learn


TF-IDF Vectorizer


Logistic Regression


What I learned
Even a small and simple dataset can give good results.

TF-IDF helped focus on the important words.

Logistic Regression worked well for this basic project.

Future Ideas
Use a bigger dataset like IMDb or Amazon reviews.

Try other models like SVM or even deep learning (like BERT).
