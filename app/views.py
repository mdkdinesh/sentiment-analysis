import os
from io import StringIO
import xlsxwriter
from django.http import HttpResponse
from django.shortcuts import render
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from string import punctuation
import spacy
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
from .models import Pos_Freq, Table_data

pd.set_option("display.max_colwidth", 200)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Globals
us_state_to_abbrev = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
    'DC': 'District of Columbia',
    'AS': 'American Samoa',
    'GU': 'Guam',
    'MP': 'Northern Mariana Islands',
    'PR': 'Puerto Rico',
    'UM': 'United States Minor Outlying Islands',
    'VI': 'U.S. Virgin Islands'}
add_stop = ["2", "26", "'s", ".", "i", "I", "��", "say", "me", "the", "my", "myself", "we", "theword", "our", "ours",
            "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
            "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
            "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
            "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "good", "should", "now"]
stop_words = set(stopwords.words('english') + list(punctuation) + list(add_stop))
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
to_freq = []


# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new


def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    return d


def get_reviews(request, city_name):
    print("in get_reviews")

    df = pd.read_excel("Feedback.xlsx", sheet_name="Feedback Details")
    df = df[df['Feedback'] != "--"].reset_index(drop=True)

    # replace "n't" with " not"
    df['Feedback'] = df['Feedback'].str.replace("n\'t", " not")

    # remove unwanted characters, numbers and symbols
    df['Feedback'] = df['Feedback'].str.replace("[^a-zA-Z#]", " ")

    # remove short words (length < 3)
    df['Feedback'] = df['Feedback'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

    # remove stopwords from the text
    df = df.loc[df['City'] == city_name]
    reviews = [remove_stopwords(r.lower().split()) for r in df['Feedback']]
    tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
    print("Data Cleaning....")

    def lemmatization(texts):
        output = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            output.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ']])
        return output

    reviews_2 = lemmatization(tokenized_reviews)
    print("Data lemmatization....")

    reviews_3 = []
    for i in range(len(reviews_2)):
        reviews_3.append(' '.join(reviews_2[i]))
    df['reviews'] = reviews_3

    # Sentiment Analysis and finding polarity
    analyzer = SentimentIntensityAnalyzer()
    df['Negative_Score'] = df['reviews'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
    df['Neutral_Score'] = df['reviews'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
    df['Positive_Score'] = df['reviews'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
    df['Compound_Score'] = df['reviews'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    print("Data Sentiment Analysis....")
    # Review Categories
    df.loc[df['Compound_Score'] > 0.2, "Sentiment"] = "Positive"
    df.loc[(df['Compound_Score'] >= -0.2) & (df['Compound_Score'] <= 0.2), "Sentiment"] = "Neutral"
    df.loc[df['Compound_Score'] < -0.2, "Sentiment"] = "Negative"

    # Percentages of Sentiment Calculation
    pos_review = round((len(df[df["Sentiment"] == "Positive"]) / len(df)) * 100)
    neu_review = round((len(df[df["Sentiment"] == "Neutral"]) / len(df)) * 100)
    neg_review = round((len(df[df["Sentiment"] == "Negative"]) / len(df)) * 100)
    print("Data compound Analysis....", pos_review, neu_review, neg_review)

    df_Postive = df[df['Sentiment'] == "Positive"]
    df_Neutral = df[df['Sentiment'] == "Neutral"]
    df_Negative = df[df['Sentiment'] == "Negative"]

    # Positive_Word_Cloud_Analysis
    Positive_Word_Cloud_Analysis = ' '.join(df_Postive['reviews'])
    if len(Positive_Word_Cloud_Analysis) == 0:
        Positive_Word_Cloud_Analysis = "NoPositive"
    wordcloud = WordCloud(
        background_color="white",
        max_words=150,
        max_font_size=40,
        scale=3,
        random_state=42
    ).generate(Positive_Word_Cloud_Analysis)

    wc = plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    wcp = StringIO()
    wc.savefig(wcp, format='svg')
    wcp.seek(0)
    plt.close(wc)
    print("Data Positive word cloud....")

    # Negative_Word_Cloud_Analysis
    Negative_Word_Cloud_Analysis = ' '.join(df_Negative['reviews'])
    if len(Negative_Word_Cloud_Analysis) == 0:
        Negative_Word_Cloud_Analysis = "NoNegative"
    wordcloud = WordCloud(
        background_color="white",
        max_words=100,
        max_font_size=25,
        scale=2,
        random_state=42
    ).generate(Negative_Word_Cloud_Analysis)

    wc_n = plt.figure(figsize=(7, 4))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    wcn = StringIO()
    wc_n.savefig(wcn, format='svg')
    wcn.seek(0)
    plt.close(wc_n)
    print("Data Negative_Word_Cloud_Analysis....")

    # Review ratings
    rating_1 = df.loc[df['Star Rating'] == 1]
    rating_2 = df.loc[df['Star Rating'] == 2]
    rating_3 = df.loc[df['Star Rating'] == 3]
    rating_4 = df.loc[df['Star Rating'] == 4]
    rating_5 = df.loc[df['Star Rating'] == 5]

    r1_pos = rating_1.loc[rating_1["Sentiment"] == "Positive"]
    r1_neu = rating_1.loc[rating_1["Sentiment"] == "Neutral"]
    r1_neg = rating_1.loc[rating_1["Sentiment"] == "Negative"]
    r2_pos = rating_2.loc[rating_2["Sentiment"] == "Positive"]
    r2_neu = rating_2.loc[rating_2["Sentiment"] == "Neutral"]
    r2_neg = rating_2.loc[rating_2["Sentiment"] == "Negative"]
    r3_pos = rating_3.loc[rating_3["Sentiment"] == "Positive"]
    r3_neu = rating_3.loc[rating_3["Sentiment"] == "Neutral"]
    r3_neg = rating_3.loc[rating_3["Sentiment"] == "Negative"]
    r4_pos = rating_4.loc[rating_4["Sentiment"] == "Positive"]
    r4_neu = rating_4.loc[rating_4["Sentiment"] == "Neutral"]
    r4_neg = rating_4.loc[rating_4["Sentiment"] == "Negative"]
    r5_pos = rating_5.loc[rating_5["Sentiment"] == "Positive"]
    r5_neu = rating_5.loc[rating_5["Sentiment"] == "Neutral"]
    r5_neg = rating_5.loc[rating_5["Sentiment"] == "Negative"]

    Rating_list = [[len(r1_pos), len(r1_neu), len(r1_neg)], [len(r2_pos), len(r2_neu), len(r2_neg)],
                   [len(r3_pos), len(r3_neu), len(r3_neg)], [len(r4_pos), len(r4_neu), len(r4_neg)],
                   [len(r5_pos), len(r5_neu), len(r5_neg)]]

    star_df = pd.DataFrame([[Rating_list[0][0], Rating_list[0][1], Rating_list[0][2]],
                            [Rating_list[1][0], Rating_list[1][1], Rating_list[1][2]],
                            [Rating_list[2][0], Rating_list[2][1], Rating_list[2][2]],
                            [Rating_list[3][0], Rating_list[3][1], Rating_list[3][2]],
                            [Rating_list[4][0], Rating_list[4][1], Rating_list[4][2]]],
                           columns=['Positive', 'Neutral', 'Negative'])
    print("Review ratings Analysis....")

    date_based = pd.DataFrame(df["Review Date"].value_counts().rename_axis("Date").reset_index()).drop("Review Date",
                                                                                                       axis=1)
    date_based = date_based.merge(df_Postive["Review Date"].value_counts().rename_axis("Date"), on="Date", how="outer")
    date_based.columns = ['Date', 'Positive']
    date_based = date_based.merge(df_Neutral["Review Date"].value_counts().rename_axis("Date"), on="Date", how="outer")
    date_based.columns = ['Date', 'Positive', 'Neutral']
    date_based = date_based.merge(df_Negative["Review Date"].value_counts().rename_axis("Date"), on="Date", how="outer")
    date_based.columns = ['Date', 'Positive', 'Neutral', 'Negative']
    date_based = date_based.fillna("0")
    for value in ['Positive', 'Neutral', 'Negative']:
        date_based[value] = [int(x) for x in date_based[value]]
    date_based = date_based.sort_values(by="Date").reset_index(drop=True)

    print("Found the date DF")

    subset = df[df['Sentiment'] == "Positive"]
    dist_pos = list(subset["Compound_Score"])
    star_pos = list(star_df["Positive"])
    star_neu = list(star_df["Neutral"])
    star_neg = list(star_df["Negative"])
    dates = list(date_based["Date"])
    date_pos = list(date_based["Positive"])
    date_neu = list(date_based["Neutral"])
    date_neg = list(date_based["Negative"])
    print("Storing the values")

    results = {
        'wcp': wcp.getvalue(),
        'wcn': wcn.getvalue(),
        'pos_count': len(df[df["Sentiment"] == "Positive"]),
        'pos_per': pos_review,
        'neu_count': len(df[df["Sentiment"] == "Neutral"]),
        'neu_per': neu_review,
        'neg_count': len(df[df["Sentiment"] == "Negative"]),
        'neg_per': neg_review,
        'dist_pos': dist_pos,
        'star_pos': star_pos,
        'star_neu': star_neu,
        'star_neg': star_neg,
        'dates': dates,
        'date_pos': date_pos,
        'date_neu': date_neu,
        'date_neg': date_neg,
        'return_name': city_name,

    }

    return results


df = pd.read_excel("Feedback.xlsx", sheet_name="Feedback Details")
df = df[df['Feedback'] != "--"].reset_index(drop=True)

# replace "n't" with " not"
df['Feedback'] = df['Feedback'].str.replace("n\'t", " not")

# remove unwanted characters, numbers and symbols
df['Feedback'] = df['Feedback'].str.replace("[^a-zA-Z#]", " ")

# remove short words (length < 3)
df['Feedback'] = df['Feedback'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

# remove stopwords from the text
reviews = [remove_stopwords(r.lower().split()) for r in df['Feedback']]
tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())

print("Data Cleaning....")


def lemmatization(texts):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ']])
    return output


reviews_2 = lemmatization(tokenized_reviews)
print("Data lemmatization....")

reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))
df['reviews'] = reviews_3

# Sentiment Analysis and finding polarity
analyzer = SentimentIntensityAnalyzer()
df['Negative_Score'] = df['reviews'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
df['Neutral_Score'] = df['reviews'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
df['Positive_Score'] = df['reviews'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
df['Compound_Score'] = df['reviews'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

print("Data Sentiment Analysis....")
# Review Categories
df.loc[df['Compound_Score'] > 0.2, "Sentiment"] = "Positive"
df.loc[(df['Compound_Score'] >= -0.2) & (df['Compound_Score'] <= 0.2), "Sentiment"] = "Neutral"
df.loc[df['Compound_Score'] < -0.2, "Sentiment"] = "Negative"


def index(request):
    if request.method == "POST":
        try:
            print(request.POST['city_name'])
            if request.POST['city_name']:
                city_name = request.POST['city_name']
                return render(request, "index.html", get_reviews(request, city_name))
        except:
            print("error.......")
        try:
            print(request.POST['state_name'])
        except:
            pass
    # get_reviews(request, "Jacksonville")

    writer = pd.ExcelWriter('Dine_Brand.xlsx', engine='xlsxwriter')

    df.to_excel(writer, sheet_name="Dataset", index=False)

    # Percentages of Sentiment Calculation
    pos_review = [j for i, j in enumerate(df['Feedback']) if df['Compound_Score'][i] > 0.2]
    neu_review = [j for i, j in enumerate(df['Feedback']) if 0.2 >= df['Compound_Score'][i] >= -0.2]
    neg_review = [j for i, j in enumerate(df['Feedback']) if df['Compound_Score'][i] < -0.2]

    df_Postive = df[df['Sentiment'] == "Positive"]
    df_Neutral = df[df['Sentiment'] == "Neutral"]
    df_Negative = df[df['Sentiment'] == "Negative"]
    df_Postive.to_excel(writer, sheet_name="Positive Reviews", index=False)
    df_Neutral.to_excel(writer, sheet_name="Neutral Reviews", index=False)
    df_Negative.to_excel(writer, sheet_name="Negative Reviews", index=False)

    # Positive_Word_Cloud_Analysis
    Positive_Word_Cloud_Analysis = ' '.join(df_Postive['reviews'])
    wordcloud = WordCloud(
        background_color="white",
        max_words=150,
        max_font_size=40,
        scale=3,
        random_state=42
    ).generate(Positive_Word_Cloud_Analysis)

    wc = plt.figure(figsize=(7, 4))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    wcp = StringIO()
    wc.savefig(wcp, format='svg')
    wcp.seek(0)
    plt.close(wc)

    # Negative_Word_Cloud_Analysis
    Negative_Word_Cloud_Analysis = ' '.join(df_Negative['reviews'])
    wordcloud = WordCloud(
        background_color="white",
        max_words=100,
        max_font_size=25,
        scale=2,
        random_state=42
    ).generate(Negative_Word_Cloud_Analysis)

    wc_n = plt.figure(figsize=(7, 4))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    wcn = StringIO()
    wc_n.savefig(wcn, format='svg')
    wcn.seek(0)
    plt.close(wc_n)
    print("Data Word_Cloud_Analysis....")

    # Review ratings
    rating_1 = df.loc[df['Star Rating'] == 1]
    rating_2 = df.loc[df['Star Rating'] == 2]
    rating_3 = df.loc[df['Star Rating'] == 3]
    rating_4 = df.loc[df['Star Rating'] == 4]
    rating_5 = df.loc[df['Star Rating'] == 5]

    r1_pos = rating_1.loc[rating_1["Sentiment"] == "Positive"]
    r1_neu = rating_1.loc[rating_1["Sentiment"] == "Neutral"]
    r1_neg = rating_1.loc[rating_1["Sentiment"] == "Negative"]
    r2_pos = rating_2.loc[rating_2["Sentiment"] == "Positive"]
    r2_neu = rating_2.loc[rating_2["Sentiment"] == "Neutral"]
    r2_neg = rating_2.loc[rating_2["Sentiment"] == "Negative"]
    r3_pos = rating_3.loc[rating_3["Sentiment"] == "Positive"]
    r3_neu = rating_3.loc[rating_3["Sentiment"] == "Neutral"]
    r3_neg = rating_3.loc[rating_3["Sentiment"] == "Negative"]
    r4_pos = rating_4.loc[rating_4["Sentiment"] == "Positive"]
    r4_neu = rating_4.loc[rating_4["Sentiment"] == "Neutral"]
    r4_neg = rating_4.loc[rating_4["Sentiment"] == "Negative"]
    r5_pos = rating_5.loc[rating_5["Sentiment"] == "Positive"]
    r5_neu = rating_5.loc[rating_5["Sentiment"] == "Neutral"]
    r5_neg = rating_5.loc[rating_5["Sentiment"] == "Negative"]

    Rating_list = [[len(r1_pos), len(r1_neu), len(r1_neg)], [len(r2_pos), len(r2_neu), len(r2_neg)],
                   [len(r3_pos), len(r3_neu), len(r3_neg)], [len(r4_pos), len(r4_neu), len(r4_neg)],
                   [len(r5_pos), len(r5_neu), len(r5_neg)]]

    star_df = pd.DataFrame([[Rating_list[0][0], Rating_list[0][1], Rating_list[0][2]],
                            [Rating_list[1][0], Rating_list[1][1], Rating_list[1][2]],
                            [Rating_list[2][0], Rating_list[2][1], Rating_list[2][2]],
                            [Rating_list[3][0], Rating_list[3][1], Rating_list[3][2]],
                            [Rating_list[4][0], Rating_list[4][1], Rating_list[4][2]]],
                           columns=['Positive', 'Neutral', 'Negative'])

    star_df_csv = star_df
    star_df_csv["Category"] = ["Star 1", "Star 2", "Star 3", "Star 4", "Star 5"]
    star_df_csv = star_df_csv.reindex(columns=['Category', 'Positive', 'Neutral', 'Negative'])
    star_df_csv.to_excel(writer, sheet_name="Star Rating", index=False)
    p = [int(x) for x in star_df["Positive"]]
    print(p)

    """workbook = writer.book
    star_chart = workbook.add_chart({'type': 'column'})
    star_chart.add_series({'values': '=Star Rating!$B$2:$B$8'})
    star_chart.add_series({'values': 'list(star_df["Neutral"])'})
    star_chart.add_series({'values': 'list(star_df["Negative"])'})

    worksheet = writer.sheets['Star Rating']
    worksheet.insert_chart('Star Rating', star_chart)"""

    pos_state_city = df.loc[df['Sentiment'] == "Positive"]
    neg_state_city = df.loc[df['Sentiment'] == "Negative"]
    top_pos_state = pd.DataFrame(
        pos_state_city["State"].value_counts().rename_axis('State').reset_index(name='Counts').head(5))
    top_pos_state['State'] = top_pos_state['State'].replace(us_state_to_abbrev)
    top_neg_state = pd.DataFrame(
        neg_state_city["State"].value_counts().rename_axis('State').reset_index(name='Counts').head(5))
    top_neg_state['State'] = top_neg_state['State'].replace(us_state_to_abbrev)
    top_pos_city = pd.DataFrame(
        pos_state_city["City"].value_counts().rename_axis('City').reset_index(name='Counts').head(5))
    top_neg_city = pd.DataFrame(
        neg_state_city["City"].value_counts().rename_axis('City').reset_index(name='Counts').head(5))

    top_pos_state.to_excel(writer, sheet_name="Top Positive States", index=False)
    top_neg_state.to_excel(writer, sheet_name="Top Negative States", index=False)
    top_pos_city.to_excel(writer, sheet_name="Top Positive City", index=False)
    top_neg_city.to_excel(writer, sheet_name="Top Negative City", index=False)

    date_based = pd.DataFrame(df["Review Date"].value_counts().rename_axis("Date").reset_index()).drop("Review Date",
                                                                                                       axis=1)
    date_based["Positive"] = list(df_Postive["Review Date"].value_counts())
    date_based["Neutral"] = list(df_Neutral["Review Date"].value_counts())
    date_based["Negative"] = list(df_Negative["Review Date"].value_counts())
    date_based = date_based.sort_values(by="Date").reset_index(drop=True)

    date_based.to_excel(writer, sheet_name="Date Based reviews", index=False)

    subset = df[df['Sentiment'] == "Positive"]
    dist_pos = list(subset["Compound_Score"])
    star_pos = list(star_df["Positive"])
    star_neu = list(star_df["Neutral"])
    star_neg = list(star_df["Negative"])
    dates = list(date_based["Date"])
    date_pos = list(date_based["Positive"])
    date_neu = list(date_based["Neutral"])
    date_neg = list(date_based["Negative"])
    print("Review ratings Analysis....")

    # Freq Positive
    positive_words = freq_words(df_Postive['reviews'], 20)
    pos_word_list = list(positive_words['word'])
    pos_freq_list = join_for_review(pos_word_list, df_Postive['reviews'])

    reviews_21 = lemmatization_noun(pd.Series(list(positive_words["word"])).apply(lambda x: x.split()))
    reviews_freq = []
    for i in range(len(reviews_21)):
        if len(reviews_21[i]) != 0:
            reviews_freq.append(reviews_21[i][0])
    to_merge = pd.DataFrame(reviews_freq)
    to_merge.columns = ["word"]
    pos_freq = positive_words.merge(to_merge, on="word", how="inner")

    pos_freq.to_excel(writer, sheet_name="Top positive keywords", index=False)

    # Freq Negative
    negative_words = freq_words(df_Negative['reviews'], 20)

    reviews_21_neg = lemmatization_noun(pd.Series(list(negative_words["word"])).apply(lambda x: x.split()))
    reviews_freq_neg = []
    for i in range(len(reviews_21_neg)):
        if len(reviews_21_neg[i]) != 0:
            reviews_freq_neg.append(reviews_21_neg[i][0])
    to_merge_neg = pd.DataFrame(reviews_freq_neg)
    to_merge_neg.columns = ["word"]
    neg_freq = negative_words.merge(to_merge_neg, on="word", how="inner")

    neg_freq.to_excel(writer, sheet_name="Top negative keywords", index=False)

    for x in pos_freq_list:
        to_freq.append(x)

    word = [x for x in pos_freq['word']]
    count = [x for x in pos_freq['count']]
    (Pos_Freq.objects.all()).delete()
    for i in range(len(pos_freq)):
        Pos_Freq.objects.create(
            word=word[i],
            freq=count[i],
        )

    writer.close()

    return render(request, "index.html", {
        'wcp': wcp.getvalue(),
        'wcn': wcn.getvalue(),
        'pos_count': len(pos_review),
        'pos_per': int(round(len(pos_review) * 100 / len(df['Feedback']))),
        'neu_count': len(neu_review),
        'neu_per': int(round(len(neu_review) * 100 / len(df['Feedback']))),
        'neg_count': len(neg_review),
        'neg_per': int(round(len(neg_review) * 100 / len(df['Feedback']))),
        'top_pos_state': top_pos_state.values,
        'top_neg_state': top_neg_state.values,
        'top_pos_city': top_pos_city.values,
        'top_neg_city': top_neg_city.values,
        'dist_pos': dist_pos,
        'star_pos': star_pos,
        'star_neu': star_neu,
        'star_neg': star_neg,
        'dates': dates,
        'date_pos': date_pos,
        'date_neu': date_neu,
        'date_neg': date_neg,
        'table_ok': 1,
        'pos_freq_word': list(pos_freq['word'])[0:7],
        'pos_freq_count': list(pos_freq['count'])[0:7],
        'neg_freq_word': list(neg_freq['word'])[0:7],
        'neg_freq_count': list(neg_freq['count'])[0:7],

    })


def lemmatization_noun(texts):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in ['NOUN']])
    return output


def charts_sample(request):
    return render(request, "chart.html")


def freq(request):
    return render(request, "freq.html", {
        "top_pos_words": Pos_Freq.objects.all(),
    })


def join_for_review(word_list, reviews):
    rev_new = []
    for x in word_list:
        in_rev = [x]
        for line in reviews:
            x = x.lower()
            line = line.lower()
            if x in line:
                in_rev.append(line)
        rev_new.append(in_rev)
    return rev_new


def dataset(request):
    table_data = pd.read_csv("table_data.csv")
    print(table_data.columns)
    date = list(table_data['Review Date'])
    city = list(table_data['City'])
    star = list(table_data['Star Rating'])
    review = list(table_data['Feedback'])
    sentiment = list(table_data['Sentiment'])

    (Table_data.objects.all()).delete()
    count = 1
    for i in range(len(sentiment)):
        if count < 21:
            count += 1
            Table_data.objects.create(
                date=date[i],
                city=city[i],
                star=star[i],
                review=review[i][0:30]+"...",
                sentiment=sentiment[i],
            )
        else:
            break
    records = {"records": Table_data.objects.all()}
    return render(request, "dataset.html", records)


def total_dataset(request):
    response = HttpResponse(open('table_data.csv', 'rb').read(), content_type='text/csv')
    response['Content-Length'] = os.path.getsize('table_data.csv')
    response['Content-Disposition'] = 'attachment; filename=%s' % 'table_data.csv'
    return response


def welcome(request):
    return render(request, "welcome.html")
