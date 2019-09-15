# -*- coding: utf-8 -*-

"""

Created on Thu Apr 18 09:29:39 2019



@author: abalter

"""

import gensim as gensim

import numpy as np

import pandas as pd

import re

from nltk.corpus import stopwords

import nltk

import requests

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer

from nltk.stem import SnowballStemmer

import gensim

import pyLDAvis.gensim



viz = False

clean_up = True



if viz == True:

    from plotnine import *



#########################################

#

#

#             DATA PROCESSING

#

#########################################



# Read in reviews

scotch = pd.read_csv('data/scotch_reviews.csv')



# Check the data

print(scotch.head())

print()



# First column is just an index, we can remove.

print(scotch.columns)

print('\n')

print(scotch.iloc[:5, 0])

print('\n')

scotch = scotch.drop(scotch.columns[0], axis = 1)





###################### CLEAN PRICE COLUMN OF STRING

print('########### CLEANING PRICE COLUMN...')

# Examine price column

print(scotch['price'])



# Find instances where price is not a number

def is_number(s):

    try:

        float(s)

        return True

    except ValueError:

        return False

    

price_filter = pd.Series(list(map(lambda x: is_number(x), scotch['price'])))



print(scotch.price[~price_filter])



# Based on the first result, let's call a "set" = Price/4.

# Luckily they are all 60,000 so we can just replace with 15,000

clean_price_1 = pd.Series(list(map(lambda x: re.sub('[$]?\d+,\d+/set', '15000', x), scotch['price'])))

print(clean_price_1[~price_filter])



# Now remove $15,000 or, and liter (remember to multiply the /liter one by 0.75 because

# a typical bottle is 750ml) - index 576

clean_price_2 = pd.Series(list(map(lambda x: re.sub('\$15,000 or |/liter', '', x), clean_price_1)))

print(clean_price_2[~price_filter])



# Finally remove commas

clean_price_3 = pd.Series(list(map(lambda x: re.sub(',','', x), clean_price_2)))

print(clean_price_3[~price_filter])



# Multiply index 576 by 0.75

clean_price_3[576] = str(float(clean_price_3[576]) * 0.75)



# Confirm

print(clean_price_3[576])



# Test clean column for numbers

price_clean_filter = pd.Series(list(map(lambda x: is_number(x), clean_price_3)))

print(price_clean_filter.value_counts())



# Replace the column

scotch['price'] = pd.to_numeric(clean_price_3)



# Clean up

if clean_up == True:

    del(clean_price_1, clean_price_2, clean_price_3, price_clean_filter, price_filter)

print()

##################### END CLEAN PRICE COLUMN



##################### ADD YEAR COLUMN

print('########## ADDING YEAR COLUMN...')



print(scotch.loc[:10, 'name'])



# Extract year by finding value 'X year old', put nan where year is not listed

# Could also use the date (some have 19XX, or 20XX listed but it seems unreliable)

years = list(map(lambda x: re.findall('(\d+) years? old', x), scotch['name']))

years = list(map(lambda x: int(x[0]) if len(x) else np.nan, years))



# How many are missing years?

years_test = pd.Series(years)

years_filter = years_test.isna()

print(str(sum(years_test.isna())) + ' are missing year.')

print()

print('Missing year statement test')

print(scotch.name[years_filter].head(10))

print()





# Create column

scotch['year_statement'] = years



# Clean up

if clean_up == True:

    del(years, years_filter, years_test)

print()

##################### END YEAR COLUMN



##################### ABV COLUMN

print('######### ADDING ABV COLUMN...')

print(scotch.loc[:10, 'name'])



# Extract any float followed by a %

abv = list(map(lambda x: re.findall('(\d+\.?\d+?)%', x), scotch['name']))

abv = list(map(lambda x: float(x[0]) if len(x) else np.nan, abv))



# How many are missing ABV?

abv_test = pd.Series(abv)

abv_filter = abv_test.isna()

print(str(sum(abv_test.isna())) + ' are missing ABV.')

print()

print('Missing year ABV test')

print(scotch.name[abv_filter])

print()



# Add ABV column

scotch['ABV'] = abv

scotch['ABV'] = scotch['ABV'].astype(float)



# Clean up

if clean_up == True:

    del(abv, abv_filter, abv_test)

print()

##################### END ABV COLUMN



##################### ADD VINTAGE_EDITION COLUMN

print('######## ADDING VINTAGE AND EDITION COLUMNS...')

# Some are listed as vintage or some type of edition

# Make boolean column to list whether name has the word "vintage" or "edition"



vintage = pd.Series(list(map(lambda x: bool(re.search('[Vv]intage', x)), scotch['name'])))

edition = pd.Series(list(map(lambda x: bool(re.search('[Ee]dition', x)), scotch['name'])))



# How many of each?

print(str(sum(vintage)) + ' names had the word "Vintage"')

print(str(sum(edition)) + ' names had the word "Edition"')



# Make three columns, one for vintage, one for edition and one for vintage or edition

scotch['Vintage'] = vintage

scotch['Special_Edition'] = edition

scotch['Vintage_or_Edition'] = vintage | edition



# Clean up

if clean_up == True:

    del(vintage, edition)





###################### END VINTAGE_EDITION





###################### BRANDS

print('######## ADDING BRAND COLUMN...')

# Let's try and scrape the website for brand names

# If they aren't on this list, then no brand will be found

# https://www.thewhiskyexchange.com/brands/scotchwhisky





# Single Malts



def extract_brands(website):

    url = website

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    soup_str = str(soup)

    scotches = re.findall('<span class="az-item-name">([A-Za-z\s\']+)</span>', soup_str)  

    scotches = [x.lstrip().rstrip() for x in scotches]

    return(scotches)



single_malts = extract_brands("https://www.thewhiskyexchange.com/brands/scotchwhisky/40/single-malt-scotch-whisky.html")

print('Single Malt List')

print(single_malts[:10])

print()



# Holy sh** it worked

# Let's get the others



# Blended

blended = extract_brands("https://www.thewhiskyexchange.com/brands/scotchwhisky/304/blended-scotch-whisky.html")

print('Blended List')

print(blended[:10])

print()



# Blended Malt

blended_malt = extract_brands("https://www.thewhiskyexchange.com/brands/scotchwhisky/309/blended-malt-scotch-whisky.html")

print('Blended Malt List')

print(blended_malt[:10])

print()



# Grain

grain = extract_brands("https://www.thewhiskyexchange.com/brands/scotchwhisky/310/grain-scotch-whisky.html")

print('Grain List')

print(grain[:10])

print()



# Combine lists

all_brands = single_malts + blended + blended_malt + grain

all_brands_upper = [x.upper() for x in all_brands]



# Get unique

all_brands_set = sorted(list(set(all_brands_upper)))

print(all_brands_set)

# Make large regular expression

all_brands_regex = '|'.join(all_brands_set)



# Try and extract brand from name column



brands = list(map(lambda x: re.findall(all_brands_regex, re.sub('’', "'",x.upper())), scotch['name']))



brands = list(map(lambda x: str(x[0]) if len(x) else 'NA', brands))



# Add brand column

scotch['brand'] = brands



print('Could not find brand name for ' + str(sum(scotch['brand'] == 'NA')) + ' scotches.')

print()

# Clean up

if clean_up == True:

    del(all_brands, all_brands_set, all_brands_upper, all_brands_regex, blended, blended_malt, brands, grain,

        single_malts)





###################### END BRANDS



###################### MAKE QUALITATIVE SCORE COLUMN

print('######## MAKING QUALITATIVE SCORE COLUMN...')

#plt.figure()

#scotch['review.point'].plot(kind = 'hist', bins = 20, title = 'Histogram of Review Score')

if viz == True:

    fig_1 = (ggplot(scotch, aes(x = 'review.point')) + 

           geom_histogram(color = 'black', fill = 'blue', binwidth = 1)+

           ggtitle('Histogram of Review Score')+

           xlab('Review Score')+

           ylab('Count')+

           scale_x_continuous(breaks = range(0,100,5)))

    print(fig_1)

# Let use the 25th percentile to be Below Average

# And the 75th percentile to be Above Average

# And between the 25th and 75th to be Average

print('25th and 7th Percentiles of Review Score')

print(scotch['review.point'].quantile([0.25, 0.75]))

print()



scotch['Review_Score_Class'] = pd.cut(scotch['review.point'], [0, 84.0, 90.0, 100],

      right = False,labels = ['Below Average', 'Average', 'Above Average'])



print(scotch['Review_Score_Class'].value_counts())

print()



##################### END QUALITATIVE SCORE COLUMN





##################### ALTER REVIEW COLUMNS



# Remove punctation and numbers and make everything lowercase

def clean_punct_numbers(text):

    # There are a few lines with \r\n

    # So that will be removed (and used to split the text later) before the numbers.

    text_new = re.sub('\r\n', ' ', text)

    # Remove punctuation

    text_new = re.sub("\.|!|\?|\(|\)|,|;", '', text_new)

    token_text = text_new.split(' ')

    new_text = [word for word in token_text if not re.match('^.*\d+.*$', word)]

    # make lowercase

    new_text = [word.lower() for word in new_text]

    new_text_string = ' '.join(new_text)

    return(new_text_string)



scotch['no_punct_numbers'] = pd.Series(list(map(lambda x: clean_punct_numbers(x), scotch['description'])))



# Let's compare one which had numbers at the very end.

print(scotch.loc[2209, 'description'])

print('\n')

print(scotch.loc[2209, 'no_punct_numbers'])

print('\n')



# Remove stopwords and make everything lowercase

stopWords = set(stopwords.words('english'))

def remove_stopwords(text):

    tokens = word_tokenize(text)

    no_stopwords = [word.lower() for word in tokens if word.lower() not in stopWords]

    string = ' '.join(no_stopwords)

    return(string)

    

scotch['no_stopwords'] = list(map(lambda x: remove_stopwords(x), scotch['no_punct_numbers']))



# Check it

print(scotch.loc[2209, 'no_punct_numbers'])

print('\n')

print(scotch.loc[2209, 'no_stopwords'])

print('\n')



# Lets also add stemming



# Porter

ps = PorterStemmer()



def porter_stem_text(text):

    tokenized = word_tokenize(text)

    stemmed_text = [ps.stem(word) for word in tokenized]

    return(' '.join(stemmed_text))

    

stemmed_reviews = list(map(lambda x: porter_stem_text(x), scotch['no_stopwords']))

scotch['porter_stemmed'] = stemmed_reviews



# Check it

print(scotch.loc[2209, 'no_stopwords'])

print('\n')

print(scotch.loc[2209, 'porter_stemmed'])

print('\n')



# Snowball

sb = SnowballStemmer('english')



def snowball_stem_text(text):

    tokenized = word_tokenize(text)

    stemmed_text = [sb.stem(word) for word in tokenized]

    return(' '.join(stemmed_text))

    

stemmed_reviews = list(map(lambda x: snowball_stem_text(x), scotch['no_stopwords']))

scotch['snowball_stemmed'] = stemmed_reviews



# Check it

print(scotch.loc[2209, 'no_stopwords'])

print('\n')

print(scotch.loc[2209, 'snowball_stemmed'])

print('\n')





word_counts_raw = pd.Series([len(x.split(' ')) for x in scotch['description']])

word_counts_no_punct_num = pd.Series([len(x.split(' ')) for x in scotch['no_punct_numbers']])

word_counts_no_stops = pd.Series([len(x.split(' ')) for x in scotch['no_stopwords']])



def mean_word_length(x):

    y = x.split(' ')

    z = [len(w) for w in y]

    return(np.mean(z))

    

mean_word_lengths = pd.Series(list(map(lambda x: mean_word_length(x), scotch['no_punct_numbers'])))

mean_word_length_all = round(mean_word_lengths.mean())

print('Mean Word Length')

print(str(mean_word_length_all) + ' characters.')

print()



stats_df = pd.DataFrame()

stats_df['Raw_Word_Counts'] = word_counts_raw

stats_df['No_Punct_Num'] = word_counts_no_punct_num

stats_df['No_Stopwords'] = word_counts_no_stops

if viz == True:

    fig_2 = (ggplot(stats_df)+

             geom_histogram(aes(x = 'Raw_Word_Counts'), fill = 'red', color = 'black')+

             xlab('Word Counts')+

             ylab('Count')+

             ggtitle('Original Word Counts')+

             scale_x_continuous(limits = [0,250])

             )

    print(fig_2)

    

    fig_3 = (ggplot(stats_df)+

             geom_histogram(aes(x = 'No_Stopwords'), fill = 'green', color = 'black')+

             xlab('Word Counts')+

             ylab('Count')+

             ggtitle('Stop Words Removed Word Counts')+

             scale_x_continuous(limits = [0,250])

             )

    print(fig_3)



# Clean up

if clean_up == True:

    del(stemmed_reviews, mean_word_lengths, stats_df, word_counts_no_punct_num, 

        word_counts_no_stops, word_counts_raw)







lda_df = scotch[['brand', 'Review_Score_Class', 'porter_stemmed']]



# settings for LDA

numtopics = 2

passes = 10

minprob = 0.1

modelname = '2topic_whiskey.model'



# Regex Tokenizer initalizer

rTokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')



docs = lda_df['porter_stemmed'].tolist()



# Tokenize the doc and remove stop words

tokens = []

for doc in docs:

    tokens.append(rTokenizer.tokenize(doc))



lda_df.insert(loc=0, column='tokens', value=tokens)



# # Generated doc-term matrix

dictionary = gensim.corpora.Dictionary(tokens)

#

# # Create bag of words

corpus = [dictionary.doc2bow(word) for word in tokens]

#

# create LDA model

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=numtopics, id2word=dictionary, passes=passes, minimum_probability=minprob)



# Save model for later loading

ldamodel.save('C:\\Git\\IST736\\Project\\data\\' + modelname)



# Load LDA model

#ldamodel = gensim.models.LdaModel.load('C:\\Git\\IST736\\Project\\data\\' + modelname)



# Show topics of model

#print(ldamodel.show_topics())



# function. For each document, show which the dominant topic is, what the percentage contribution is, and keywords

def format_topics_sentences(ldamodel, texts, corpus):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row_list in enumerate(ldamodel[corpus]):

        row = row_list[0] if ldamodel.per_word_topics else row_list

        # print(row)

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)



# Show all columns and do not truncate in a DF

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', -1)



# create the df with the topics and documents

df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=lda_df['porter_stemmed'].tolist())

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

print(df_dominant_topic.head(10))





# Generate a wordcloud

if viz == True:

    from matplotlib import pyplot as plt

    from wordcloud import WordCloud

    import matplotlib.colors as mcolors



    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



    cloud = WordCloud(stopwords=stopWords,

                      background_color='white',

                      width=2500,

                      height=2000,

                      max_words=100,

                      colormap='tab10',

                      color_func=lambda *args, **kwargs: cols[i],

                      prefer_horizontal=1.0)



    topics = ldamodel.show_topics(formatted=False)



    fig, axes = plt.subplots(numtopics, 1, figsize=(60, 20 * numtopics), sharex=True, sharey=True)



    for i, ax in enumerate(axes.flatten()):

        fig.add_subplot(ax)

        topic_words = dict(topics[i][1])

        print(topic_words)

        cloud.generate_from_frequencies(topic_words, max_font_size=300)

        plt.gca().imshow(cloud)

        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))

        plt.gca().axis('off')





    plt.subplots_adjust(wspace=0, hspace=0)

    plt.axis('off')

    plt.margins(x=0, y=0)

    plt.tight_layout()

    plt.show()

	

## Create Initial Wordclouds Based Upon All and Category

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt



print('Overall WordCloud')



text = scotch.description.values



wordcloud = WordCloud(

   width = 3000,

   height = 2000,

   background_color = 'white',

   ).generate(str(text))



#wordcloud = WordCloud().generate(str(text)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

#plt.tight_layout(pad=0)

plt.show()



#######################################################################################



BMSW = scotch[scotch['category'] == 'Blended Malt Scotch Whisky']

BSW = scotch[scotch['category'] == 'Blended Scotch Whisky']

GSW = scotch[scotch['category'] == 'Grain Scotch Whisky']

SGW = scotch[scotch['category'] == 'Single Grain Whisky']

SMS = scotch[scotch['category'] == 'Single Malt Scotch']







SMS = scotch[scotch['category'] == 'Single Malt Scotch']

print('Single Malt Scotch WordCloud')



#text = scotch.description.values

text = SMS.description.values



wordcloud = WordCloud(

   width = 3000,

   height = 2000,

   background_color = 'white',

   ).generate(str(text))



#wordcloud = WordCloud().generate(str(text)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

#plt.tight_layout(pad=0)

plt.show()



#######################################################################################

SGW = scotch[scotch['category'] == 'Single Grain Whisky']

print('Single Grain Whisky WordCloud')



#text = scotch.description.values

text = SGW.description.values



wordcloud = WordCloud(

   width = 3000,

   height = 2000,

   background_color = 'white',

   ).generate(str(text))



#wordcloud = WordCloud().generate(str(text)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

#plt.tight_layout(pad=0)

plt.show()



#######################################################################################

GSW = scotch[scotch['category'] == 'Grain Scotch Whisky']

print('Grain Scotch Whisky WordCloud')



#text = scotch.description.values

text = GSW.description.values



wordcloud = WordCloud(

   width = 3000,

   height = 2000,

   background_color = 'white',

   ).generate(str(text))



#wordcloud = WordCloud().generate(str(text)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

#plt.tight_layout(pad=0)

plt.show()



#######################################################################################

BSW = scotch[scotch['category'] == 'Blended Scotch Whisky']

print('Blended Scotch Whisky WordCloud')



#text = scotch.description.values

text = BSW.description.values



wordcloud = WordCloud(

   width = 3000,

   height = 2000,

   background_color = 'white',

   ).generate(str(text))



#wordcloud = WordCloud().generate(str(text)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

#plt.tight_layout(pad=0)

plt.show()



#######################################################################################

BMSW = scotch[scotch['category'] == 'Blended Malt Scotch Whisky']

print('Blended Malt Scotch Whisky WordCloud')



#text = scotch.description.values

text = BMSW.description.values



wordcloud = WordCloud(

   width = 3000,

   height = 2000,

   background_color = 'white',

   ).generate(str(text))



#wordcloud = WordCloud().generate(str(text)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

#plt.tight_layout(pad=0)

plt.show()





############################################################

## Machine Learning

## Jonah Witt

############################################################



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

import six

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO

from IPython.display import Image

import pydotplus



# This is a function to display data frames as a table

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,

                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',

                     bbox=[0, 0, 1, 1], header_columns=0,

                     ax=None, **kwargs):

    if ax is None:

        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])

        fig, ax = plt.subplots(figsize=size)

        ax.axis('off')

    

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)



mpl_table.auto_set_font_size(False)

mpl_table.set_fontsize(font_size)



for k, cell in  six.iteritems(mpl_table._cells):

    cell.set_edgecolor(edge_color)

    if k[0] == 0 or k[1] < header_columns:

        cell.set_text_props(weight='bold', color='w')

        cell.set_facecolor(header_color)

        else:

            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    return ax



# This is code for plotting confusion matrices

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(target_names))

    plt.xticks(tick_marks, target_names, rotation=45)

    plt.yticks(tick_marks, target_names)

    plt.tight_layout()

    

    width, height = cm.shape

    

    for x in range(width):

        for y in range(height):

            plt.annotate(str(cm[x][y]), xy=(y, x),

                         horizontalalignment='center',

                         verticalalignment='center')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



#This is a function for plotting most important features

def plot_coefficients(classifier, feature_names, top_features=10):

    coef = classifier.coef_.ravel()

    top_positive_coefficients = np.argsort(coef)[-top_features:]

    top_negative_coefficients = np.argsort(coef)[:top_features]

    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # create plot

    plt.figure(figsize=(15, 5))

    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]

    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)

    feature_names = np.array(feature_names)

    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')

    plt.show()



# Tokenize df

my_vectorizer = CountVectorizer(input='contents',

                                analyzer = 'word',

                                stop_words='english',

                                lowercase = True

                                )



transformed_vectors = my_vectorizer.fit_transform(scotch.snowball_stemmed)

column_names = my_vectorizer.get_feature_names()

vectorized_dtm = pd.DataFrame(transformed_vectors.toarray(),columns = column_names)



# Normalize df

my_vectorizer = TfidfVectorizer(

                                input='contents',

                                analyzer = 'word',

                                stop_words='english',

                                lowercase = True

                                )



transformed_vectors = my_vectorizer.fit_transform(scotch.snowball_stemmed)

column_names = my_vectorizer.get_feature_names()

normalized_dtm = pd.DataFrame(transformed_vectors.toarray(),columns = column_names)



# Get Sentiment df

sentiment_df = pd.DataFrame()

sentiment_df = sentiment_df.append(normalized_dtm)

sentiment = scotch['review.point']

sentiment_list = sentiment.tolist()



# Convert Sentiment to int

sentiment_list = list(map(int, sentiment_list))



# Get sentiment labels

sentiment_list = ["positive" if i >= 85 else "negative" for i in sentiment_list]

sentiment_df['label'] = sentiment_list



# Get Price df



price = scotch['price']

price_df = pd.DataFrame()

price_df = price_df.append(normalized_dtm)

price_list = price.tolist()



# Convert price to int

prie_list = list(map(int, price_list))



# Get price labels

price_list = ["expensive" if i >= 110 else "cheap" for i in price_list]

price_df['label'] = price_list



# Naive Bayes



def my_nb_results(df):

    # Split the data

    train_df, test_df = train_test_split(df, test_size=0.1)

    

    # Get test labels

    test_labels = test_df['label']

    

    # Remove test labels

    test_df = test_df.drop(['label'], axis=1)

    

    # Create nb modeler

    my_nb = MultinomialNB()

    

    # Separate labels from train data

    train_no_labels = train_df.drop(['label'], axis=1)

    train_labels = train_df['label']

    

    # Fit the svm model

    my_nb.fit(train_no_labels, train_labels)

    predictions = my_nb.predict(test_df)

    

    # Get the confusion matrix

    cnf_matrix = confusion_matrix(test_labels, predictions)

    print("The confusion matrix is:")

    print(cnf_matrix)

    plot_confusion_matrix(cnf_matrix, ['0','1'])

    

    

    # Compute the ten fold cross validation

    # print("The 10-fold cross validation is")

    # print(cross_val_score(my_svm, train_no_labels, train_labels, cv = 10))

    # cv = cross_val_score(my_svm, train_no_labels, train_labels, cv = 10)

    # cv_df = pd.DataFrame()

    # cv_df['Cross Val Scores'] = cv

    # cv_df = cv_df.round(4)

    # render_mpl_table(cv_df)

    

    # Plot feature importance

    plot_coefficients(my_nb, column_names)



# Bernoulli



def my_bernoulli_results(df):

    # Split the data

    train_df, test_df = train_test_split(df, test_size=0.1)

    

    # Get test labels

    test_labels = test_df['label']

    

    # Remove test labels

    test_df = test_df.drop(['label'], axis=1)

    

    # Create bernoulli modeler

    my_bern = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

    

    # Separate labels from train data

    train_no_labels = train_df.drop(['label'], axis=1)

    train_labels = train_df['label']

    

    # Fit the svm model

    my_bern.fit(train_no_labels, train_labels)

    predictions = my_bern.predict(test_df)

    

    # Get the confusion matrix

    cnf_matrix = confusion_matrix(test_labels, predictions)

    print("The confusion matrix is:")

    print(cnf_matrix)

    plot_confusion_matrix(cnf_matrix, ['0','1'])

    

    

    # Compute the ten fold cross validation

    # print("The 10-fold cross validation is")

    # print(cross_val_score(my_svm, train_no_labels, train_labels, cv = 10))

    # cv = cross_val_score(my_svm, train_no_labels, train_labels, cv = 10)

    # cv_df = pd.DataFrame()

    # cv_df['Cross Val Scores'] = cv

    # cv_df = cv_df.round(4)

    # render_mpl_table(cv_df)

    

    # Plot feature importance

    plot_coefficients(my_bern, column_names)





# Support Vector Machine



def my_svm_results(df):

    # Split the data

    train_df, test_df = train_test_split(df, test_size=0.1)

    

    # Get test labels

    test_labels = test_df['label']

    

    # Remove test labels

    test_df = test_df.drop(['label'], axis=1)

    

    # Create svm modeler

    my_svm = LinearSVC(C=10)

    

    # Separate labels from train data

    train_no_labels = train_df.drop(['label'], axis=1)

    train_labels = train_df['label']

    

    # Fit the svm model

    my_svm.fit(train_no_labels, train_labels)

    predictions = my_svm.predict(test_df)

    

    # Get the confusion matrix

    cnf_matrix = confusion_matrix(test_labels, predictions)

    print("The confusion matrix is:")

    print(cnf_matrix)

    plot_confusion_matrix(cnf_matrix, ['0','1'])

    

    

    # Compute the ten fold cross validation

    # print("The 10-fold cross validation is")

    # print(cross_val_score(my_svm, train_no_labels, train_labels, cv = 10))

    # cv = cross_val_score(my_svm, train_no_labels, train_labels, cv = 10)

    # cv_df = pd.DataFrame()

    # cv_df['Cross Val Scores'] = cv

    # cv_df = cv_df.round(4)

    # render_mpl_table(cv_df)

    

    # Plot feature importance

    plot_coefficients(my_svm, column_names)



# Decision Tree



def my_decision_tree_results(df):

    # Split the data

    train_df, test_df = train_test_split(df, test_size=0.10)

    

    # Get test labels

    test_labels = test_df['label']

    

    # Remove test labels

    test_df = test_df.drop(['label'], axis=1)

    

    # Create svm modeler

    my_dt = DecisionTreeClassifier()

    

    # Separate labels from train data

    train_no_labels = train_df.drop(['label'], axis=1)

    train_labels = train_df['label']

    

    # Fit the svm model

    my_dt.fit(train_no_labels, train_labels)

    predictions = my_dt.predict(test_df)

    

    # Get the confusion matrix

    cnf_matrix = confusion_matrix(test_labels, predictions)

    print("The confusion matrix is:")

    print(cnf_matrix)

    plot_confusion_matrix(cnf_matrix, ['0','1'])

    

    

    # Compute the ten fold cross validation

    # print("The 10-fold cross validation is")

    # print(cross_val_score(my_svm, train_no_labels, train_labels, cv = 10))

    # cv = cross_val_score(my_svm, train_no_labels, train_labels, cv = 10)

    # cv_df = pd.DataFrame()

    # cv_df['Cross Val Scores'] = cv

    # cv_df = cv_df.round(4)

    # render_mpl_table(cv_df)

    

    # Display tree

    dot_data = StringIO()

    export_graphviz(my_dt, out_file=dot_data,

                    filled=True, rounded=True,

                    special_characters=True)

                    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

                    graph.write_png('tree.png')

                    print(Image(graph.create_png))



my_svm_results(sentiment_df)

my_nb_results(sentiment_df)

my_bernoulli_results(sentiment_df)



my_svm_results(price_df)

my_nb_results(price_df)

my_bernoulli_results(price_df)



my_decision_tree_results(sentiment_df)

my_decision_tree_results(price_df)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

MyDTM_DF.head()
from sklearn.preprocessing import normalize
data_scaled = normalize(MyDTM_DF)
data_scaled = pd.DataFrame(data_scaled, columns=MyDTM_DF.columns)

data_scaled.head()
len(data_scaled)

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

len(data_scaled)

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=5, color='r', linestyle='--')

BMSW = scotch[scotch['category'] == 'Blended Malt Scotch Whisky']
BSW = scotch[scotch['category'] == 'Blended Scotch Whisky']
GSW = scotch[scotch['category'] == 'Grain Scotch Whisky']
SGW = scotch[scotch['category'] == 'Single Grain Whisky']
SMS = scotch[scotch['category'] == 'Single Malt Scotch']

ALL = list(scotch['snowball_stemmed'])
BMSW_reviews = list(BMSW['snowball_stemmed'])
BSW_reviews = list(BSW['snowball_stemmed'])
GSW_reviews = list(GSW['snowball_stemmed'])
SGW_reviews = list(SGW['snowball_stemmed'])
SMS_reviews = list(SMS['snowball_stemmed'])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

Sc_Cat = 'ALL'

if Sc_Cat == 'BMSW':
    documents = BMSW_reviews
elif Sc_Cat == 'BSW':
    documents = BSW_reviews
elif Sc_Cat == 'SGW':
    documents = SGW_reviews
elif Sc_Cat == 'SMS':
    documents = SMS_reviews
elif Sc_Cat == 'GSW':
    documents = GSW_reviews
else:
    documents = ALL
    


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
print(X.shape)


#labels = documents.attrs

terms = vectorizer.get_feature_names()


from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(X)
cos = cosine_similarity(X)
print(dist)
print(cos)


# Data Preprocessing

# Importing the Library 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


#Using the elbow method to find the optimal number of clusters 
from sklearn.cluster import KMeans
wcss = []
i = float()
for i in range(1, 21):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 100, n_init = 20, random_state = 0)
    kmeans.fit(X)
    wcss.append (kmeans.inertia_)
plt.plot(range(1, 21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Run the Kmeans algorithm and get the index of data points clusters
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

# Data Preprocessing
import cmath as math
import sys
# Applying k-means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

clusters = model.labels_.tolist()


print("Category",Sc_Cat)
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print(" ")
    print("Cluster %d:" % i, end=" "),
    for ind in order_centroids[i, :5]:
        print(' %s' % terms[ind], end=" "),
    print(" ")
#    print("Cluster %d titles:" % i, end='')
#    for title in frame.ix[i]['title'].values.tolist():
#        print(' %s,' % title, end='')

  

print("\n")
print("Prediction")
print(BMSW_reviews[2])
i=BMSW_reviews[2]
Y = vectorizer.transform([i])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Aromatic sherri flavors are good"])
prediction = model.predict(Y)
print(prediction)

# VISUALIZING DOCUMENT CLUSTERS

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
#cluster_names = {0: 'nose  cask  bottl', 
#                 1: 'sherri  cask  nose', 
#                 2: 'spice  finish  orang', 
#                 3: 'whiski  note  vanilla', 
#                 4: 'smoke  water  sweet'}
cluster_names = {1: 'Vanilla Fruit', 
                 3: 'Sherry Cask', 
                 4: 'Sweet Water', 
                 2: 'Scent Finish', 
                 0: 'Peat Smoke '}

len(cluster_colors)

# Multidimensional Scaling
###  TAKES A LOOOONG TIME TO RUN
import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()

#for i in range(len(documents):
titles = scotch['category']
titles

#some ipython magic to show the matplotlib plots inline
%matplotlib inline 

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 17)) 
# Original
#fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
#for i in range(len(df)):
#    ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)  

    
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)


#make 3d embedding 
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import random
mds = manifold.MDS(n_components=3, metric=True, max_iter=3000, eps=1e-9, 
                   dissimilarity="precomputed", n_jobs=1)
embed3d = mds.fit(dist).embedding_

# plot 3d embedding
#since it is a surface of constant negative curvature (hyperbolic geometry)
#expect it to look like the pseudo-sphere
#http://mathworld.wolfram.com/Pseudosphere.html
#make some random samples in 2d
# choose a different color for each point
n_samples = 20
seed = np.random.RandomState(seed=3)

#create a set of Gaussians in a grid of mean (-1.5,1.5) and standard devaition (0.2,5)
gridMuSigma=[]
for i in np.linspace(-1.5,1.5,n_samples):
    for j in np.linspace(.2,5,n_samples):
        gridMuSigma.append([i,j])
gridMuSigma=np.array(gridMuSigma)
#colors = plt.cm.jet(np.linspace(0, 1, 2247))
#colors = plt.cm.Set3(np.linspace(0, 1, 5))

#Setup plots
fig = plt.figure(figsize=(5*3,4.5))

# choose a different color for each point
n = 5
colors = plt.cm.jet(np.linspace(0,1,n))

subpl = fig.add_subplot(132,projection='3d')
subpl.scatter(embed3d[:, 0], embed3d[:, 1], embed3d[:, 2],s=20, c=cluster_colors[2])
subpl.view_init(42, 101) #looks good when njobs=-1
subpl.view_init(-130,-33)#looks good when njobs=1

plt.suptitle('3D K-Means Document Clustering')
plt.axis('tight')
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)

import mpld3
mpld3.enable_notebook()
from mpld3 import plugins

class TopToolbar(plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("y", 2);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

import numpy as np
import mpld3
from mpld3 import plugins
#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

# Plot 
fig, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                     label=cluster_names[name], mec='none', 
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]
    
    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    
    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    
ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() 
