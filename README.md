# Sarcasm Detection in Twitter Tweets


---
## Problem Statement

Will machine learning be able to detect sarcasm in Tweets posted on Twitter without the conversation's context?

---
## Background
In Natural Language Processing, sarcasm can be difficult to detect and can add additional difficulty when accessing true sentiment and meaning when analyzing text data. The use of sarcasm gives words that may mean something in one context, a completely different meaning in another context. In today's social media landscape, sarcasm often plays a big role in text communication. Whereas in-person communication benefits from tonal and bodily cues to indicate sarcasm, text may or may not adequately infuse a sarcastic tone into the narrative. Prone to misinterpretation, social media can be a particularly tricky platform to navigate should your own meter for detecting sarcasm not be aligned with the majority. As difficult as it can be for real people to identify sarcasm, language analysis methods also tend to find sarcasm difficult to detect and distinguish from heartfelt, non-sarcastic statements. In this analysis, I'll be taking two different datasets compiled from Tweets on Twitter and combining them to form a corpus consisting of the base tweets from each initial dataset. Then I will be training various machine learning models on this data to identify the presence of sarcasm. Finally I will take these models and deploy them to an app on Heroku that will allow a user to input a potential Tweet and help the user predict if the tweet will be taken sarcastically, or not.


---
## Table of Contents

```
Capstone
│   README.md
│   .gitignore    
│
└───Code
│   │   Pull_tweets.ipnyb
│   │   Cleaning_and_EDA.ipynb
│   │   Modeling.ipynb
│   │
│   └───App
│       │   file111.txt
│       │   file112.txt
│       │   ...
│   
└───Data
│   │   sarcasm_corpus.csv
│   │   model_df.csv
│   
└───Images
│   │   sentiment_histogram.png
│   │   sentiment_heatmap.png
│   │   sarcastic_sentiment_histogram.png
│   │   Number_of_emojis_used_per_tweet.png
│   │   non_sarcastic_sentiment_histogram.png
│   │   Frequency_of_Null_Values.png
│
└───Assets
│   │   
│   │   
│
..............Not included in repository.................
└───Protected
│   │
│   └───Data
│       │   corpus_processed.csv
│       │   corpus_with_text.csv
│       └───sarcasm-master

```


---
## Datasets and sources
The first dataset was created by Gavin Abercrombie, and pulled from the following [link](https://data.mendeley.com/datasets/fn2mmff85g/1). The tweets in this dataset were received having been manually scored based on the context of the tweet in the string of replies from user 1 to user 2. Only the Author tweets were used in this analysis, and were pulled using Tweepy and the Twitter API. Approximately 600 of the original tweets were no longer available. This dataset accounts for 1580 tweets  or ~20% of the Tweets used in this analysis.

The second dataset from Ghosh, S. included self-labeled tweets labeled with the hashtag "sarcasm" or "sarcastic" and a user tag indicating the tweet belongs to a larger conversation. This dataset comes from the following [repository](https://github.com/EducationalTestingService/sarcasm) and accounts for ~80% of the Tweets used in this analysis.



---
## Data Cleaning and Ingestion

To assemble the dataframe I first needed to pull the Tweet text from Twitter for the Abercrombie (2018) dataset that supplied tweet IDs. Using Tweepy I  found the tweets and assembled a dataframe from the json contained in the Tweepy status object json attribute. Several of the tweets contained return characters that contributed to formatting problems when further processing a saved csv of the tweets, so I replaced them with another character that would ensure the subsequent csv would not be misread due to the return characters included in the text.
Additionally, about 660 tweets from the original list of Tweets were no longer available, but they did not impact the positive and negative class ratios when dropped from the dataset. The missing tweets were equally distributed throughout the dataset as seen in the histogram below, and were unable to be pulled directly from the Twitter API as well. Likely, those tweets have since been deleted from the platform, or are otherwise unavailable permanently.
![Unavailable Tweets ](./Images/Frequency_of_Null_Values.png)

The second dataset used, was from Ghosh (2020) and contained 5000 Tweet texts with user handles replaced by '@USER'. Tweets were self-labeled as Sarcastic, or not Sarcastic, determined by the hashtag #Sarcasm, or #Sarcastic. Ghosh (2020) used the previous responses in the conversation to aid in their attempt to detect sarcasm.  I used just the responses column from this dataset, which is the final response to the context contained in the rest of the Tweet conversations as I also only pulled the analogous tweets from the first dataset and wanted to understand how machine learning might perform when given only the final tweet in a conversation.

The two datasets were merged to form a dataset with 8380 Tweets  that were scored either 1- Sarcastic, or 0- Not Sarcastic.


---
## Key Features

|Feature|Type|Origin Dataset|Description|
|---|---|---|---|


---
## Exploratory Data analysis
From personal experience I felt that both sentiment and emoji usage may correlate with sarcasm. To explore the impact of emoji usage on the target column I first created a list of every unique emoji present in the database, there were 336 emojis used in the text corpus and 2500 instances of emoji usage. 87% of the corpus did not include any emojis, and only 8% of the sarcastic tweets used emojis. When evaluated, the correlation between number of emojis included in a tweet and the target column of sarcasm was only -0.134.

Sentiment analysis is often brought in as a example for why sarcasm detection can be difficult in Natural Language Processing. Riloff, R. (2013) uses the example "yay! it's a holiday weekend and i'm on call for work!" to illustrate how 'yay,' typically scored as highly positive, indicates sarcasm and the opposite in the context of the full sentence.  Although this analysis will not focus on incorporating sentiment analysis techniques to make predictions, I was interested in performing a sentiment analysis on the data. I used the sentiment analyzer in NLTK's Vader package to score each tweet's sentiment.
![heatmap](./Images/sentiment_heatmap.png)


In the heatmap above we can see that the sentiment scores and sarcasm have weak relationships ranging from -0.19 (Compounded Sentiment Score) to 0.17 (Negative Sentiment Score). Further investigation indicates that 70% of the dataset scored positively when the sentiment score was compounded. In the histogram below are two series, indicating the Sarcasm/No Sarcasm label, you can see that the sarcastic tweets form a more normal distribution than the not sarcastic tweets which skewed positive. Further statistical analysis in the form of a t-test shows that the mean compounded sentiment of these two groups of tweets are not the same.
![histogram](./Images/sentiment_histogram.png)








---
## Modeling
To model the problem of predicting which tweets were sarcastic vs not sarcastic, I explored multiple classification models with either TfidfVectorizer or CountVectorizer both Natural Language Processors from Scikit-learn's feature extraction package. I found that TfidfVectorizer often produced more accurate results than CountVectorizer. In the table below are the names of models cooresponding to rows in the model_df dataframe generated during the model fitting process. As models are generated, scores and parameters are saved to a row to persist the information. Some model and transformer combinations were run with different parameter sets, but the optimal model produced by the search chose the same parameter options and produced the same scores.

For this project, I wanted to optimize the accuracy score, and the f1 score which balances false positives and false negatives. Both the Support Vector Classifier Models and Multinomial Naive-Bayes models performed well in both target metrics and were used to generate predictions in the application. Overall 56.7% of the dataset included in the model has been scored negative for sarcasm, so the baseline against which I'm measuring Accuracy is 56.7%. Every model I ran beat this score, with the two chosen models at 69.1 and 70.5% accuracy.

In each model only the text data of each tweet was considered, in future versions of this project I would be interested in exploring additional tweet features which were either unavailable for the second dataset, or not yet included due to limited scope. Although sentiment analysis was included in the initial analysis, it has not yet been incorporated into the machine learning models.

|Model|Score|f1 Score|AUC Score|Specificity (%)|Sensitivity(%)|Misclassification Rate(%)|Accuracy (%)
|---|---|---|---|---|---|---|---|
model_LogisticRegression()_TfidfVectorizer()_1|0.70|0.62|0.68|80.77|56.05|30.19|69.81
model_SVC()_TfidfVectorizer()_2|0.70|0.65|0.70|77.41|61.88|29.47|70.53
model_SVC()_TfidfVectorizer()_9|0.70|0.65|0.70|77.41|61.88|29.47|70.53
model_MultinomialNB()_CountVectorizer()_3|0.71|0.66|0.68|67.62|68.79|31.86|68.14
model_MultinomialNB()_TfidfVectorizer()_4|0.71|0.67|0.69|68.76|69.51|30.91|69.09
model_MultinomialNB()_TfidfVectorizer()_5|0.71|0.67|0.69|68.76|69.51|30.91|69.09
model_BaggingClassifier(base_estimator=RandomForestClassifier())_TfidfVectorizer()_8|0.69|0.59|0.66|81.2|51.39|32.02|67.98
model_MultinomialNB()_TfidfVectorizer()_10|0.71|0.67|0.69|68.76|69.51|30.91|69.09
model_BaggingClassifier()_TfidfVectorizer()_6|0.66|0.57|0.63|73.27|52.83|35.8|64.2
model_BaggingClassifier()_TfidfVectorizer()_7|0.68|0.56|0.65|85.06|45.65|32.42|67.58

---
## App
Because I wanted to help aid sarcasm detection for users wh omay have a hard time recognizing sarcasm in their own tone, an integral part of my project was to have a framework that could provide some feedback to a user. To do this I've built an application on Heroku with Flask that incorporates one of my machine learning models to determine if what a user has submitted is sarcastic or not. The app takes a user's input, runs the string through a processor to tokenize and lemmatize the input to get it into the same format as the training dataset. Then a model is applied and the user is taken to a result's template that reminds the user of their initial input, provides the lemmatized input string from the pre-processor, and the final prediction of Sarcastic or Not Sarcastic.


---
## Conclusions and Recommendations
While the machine learning models were often able to provide accurate sarcasm predictions, they suffered greatly from the skewed content in the twitter datasets. The majority of the initial dataset was pulled during 2020, which shows in the top content for both sarcastic and non-sarcastic data. For example, as 2020 was an election year, in the full dataset, the word Trump appears 217 times in the dataset, of which 67% are sarcastic Tweets.


---
## Sources

Riloff, R. (2013). Sarcasm as Contrast between a Positive Sentiment and Negative Situation. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (pp. 704–714). Association for Computational Linguistics.

Abercrombie, Gavin (2018), “Corpus of Sarcasm in Twitter Conversations”, Mendeley Data, V1, doi: 10.17632/fn2mmff85g.1


Ghosh, S. (2020). A Report on the 2020 Sarcasm Detection Shared Task. In Proceedings of the Second Workshop on Figurative Language Processing (pp. 1–11). Association for Computational Linguistics.




---
#Required Packages
Numpy, Pandas, Matplotlib Scikit-learn, Seaborn, Flask, NLTK, Emoji, Pickle
