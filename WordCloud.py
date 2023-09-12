import nltk
from wordcloud import WordCloud, STOPWORDS
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import re

def read_all_data_from_a_file(file_name=''):

    print('start reading file')
    f = open(file_name, 'r', errors='ignore')
    text = f.read()
    f.close()
    print('end reading file')
    return text

def ngram_convertor(sentence,n=1):
    ngram_sentence = ngrams(sentence.split(), n)
    for item in ngram_sentence:
        print(item)
    return ngram_sentence


if __name__=='__main__':
    sentence = "Life is either a daring adventure or nothing at all"
    #ngram_convertor(sentence, 2)

    WNL = nltk.WordNetLemmatizer()
    file_name = "S:\\Research-data\\Sharafkhaneh Group\\H-35366 OSA\\Data\\Respicardia\\PSG90KNotes\\Latent-analysis\\All-PSG-90k-notes.txt"
    text = read_all_data_from_a_file(file_name)
    print('Read from the file')
    # Lowercase and tokenize
    text = text.lower()
    # Remove single quote early since it causes problems with the tokenizer.
    text = text.replace("'", "")

    tokens = nltk.word_tokenize(text)
    text1 = nltk.Text(tokens)

    # Remove extra chars and remove stop words.
    text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

    print('Removed extra space')

    # set the stopwords list
    stopwords_wc = set(STOPWORDS)
    customised_words = ['from', 'subject', 'use', 'medicine','night','study','patient','per']  # If you want to remove any particular word form text which does not contribute much in meaning

    new_stopwords = stopwords_wc.union(customised_words)
    text_content = [word for word in text_content if word not in new_stopwords]

    # After the punctuation above is removed it still leaves empty entries in the list.
    text_content = [s for s in text_content if len(s) != 0]


    # Best to get the lemmas of each word to reduce the number of similar words
    #text_content = [WNL.lemmatize(t) for t in text_content]

    nltk_tokens = nltk.word_tokenize(text)
    print('Word tokenization is done')
    #full_text = ' '.join(text_content)
    bigrams_list = list(nltk.bigrams(text_content))
    #bigrams_list = list(ngrams(text_content,5))
    print('bigram_list')
    print(bigrams_list)
    dictionary2 = [' '.join(tup) for tup in bigrams_list]
    print('dictionary2')
    print(dictionary2)


    # Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(5, 5))
    bag_of_words = vectorizer.fit_transform(dictionary2)
    vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    print('word freq')
    #f_out = open('S:\\Research-data\\Sharafkhaneh Group\\H-35366 OSA\\Data\\Respicardia\\PSG90KNotes\\Latent-analysis\\bigram_top_500_word_freq.txt','w')
    print(words_freq[:500])
    print('length of word freq',len(words_freq))
    #for i in range(500):
    #    f_out.writelines(words_freq[i])
    #f_out.close()

    # Generating wordcloud and saving as jpg image
    word_cloud = 0
    if word_cloud == 1:
        words_dict = dict(words_freq)
        WC_height = 1000
        WC_width = 1500
        WC_max_words = 100
        wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
        wordCloud.generate_from_frequencies(words_dict)
        plt.title('Most frequently occurring fivegrams in PSG notes')
        plt.imshow(wordCloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        wordCloud.to_file('wordcloud_fivegram.jpg')

    f_out = open(
        "S:\\Research-data\\Sharafkhaneh Group\\H-35366 OSA\\Data\\Respicardia\\PSG90KNotes\\Latent-analysis\\Five-gramWord-Freq-PSG90k" + ".csv",
        'w')
    column_header = "FiveGramWord,Frequency\n"
    f_out.write(column_header)

    s = str(words_freq)
    new_s = re.sub('[\[*\(*\]*]+','', s)
    print('Removing [] ( from words freq')
    splitter = re.compile(r'(\),)')
    l = splitter.split(new_s)
    print('Splitting by ),')
    for i in range(len(l)):
        if l[i] != '),':
            #print('Token ', i, l[i])
            f_out.write(l[i] + '\n')
    f_out.close()