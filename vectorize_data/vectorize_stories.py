import boto3
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import json

AUTHORS = [ 'C.S. Lewis', 'Ernest Hemingway', 'Jane Austen', 'Jhumpa Lahiri', 'JK Rowling', 'Shakespeare', 'John Steinbeck',  'Bill Bryson', 'JRR Tolkein','Amy Tan']

ACCESS_KEY = ''

SECRET_ACCESS_KEY = ''
BUCKET_NAME = 'ap-math-210'
PREFIX = 'processed_data/'
VECTORIZED_PREFIX = 'vectorized_data'

s3client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_ACCESS_KEY)

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def generate_document_key_list():
    s3client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_ACCESS_KEY)

    response = s3client.list_objects_v2(Bucket = BUCKET_NAME, Prefix =PREFIX)
    documents_list = []
    key_list = []
    author_words_map = dict((author, ' ') for author in AUTHORS)
    counter = 0
    NEXT_TOKEN = ''

    while 'NextContinuationToken' in response or counter == 0:
        for object in response['Contents']:
            curr_key = object['Key']
            folder_indices = find(curr_key, '/')
            author = curr_key[folder_indices[-2]+1:folder_indices[-1]]
            #if 'Hemingway' in object['Key'] or 'Shakespeare' in object['Key'] or 'Steinbeck' in object['Key']:
            s3_response_object = s3client.get_object(Bucket=BUCKET_NAME, Key=curr_key)
            object_content = s3_response_object['Body'].read()
            author_words_map[author] = author_words_map[author] + ' ' + str(object_content)
            documents_list.append(object_content)
            key_list.append(curr_key)

        counter = counter + 1
        if 'NextContinuationToken' in response:
            NEXT_TOKEN = response['NextContinuationToken']

        response = s3client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX, ContinuationToken = NEXT_TOKEN)


    return (author_words_map, documents_list, key_list)


def vectorize_documents_and_get_tf_idf(documents_list):
    # Initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # Vectorize document using TF-IDF
    tfidf = TfidfVectorizer(lowercase=True,
                            stop_words='english',
                            ngram_range = (1,1),
                            tokenizer=tokenizer.tokenize)

    # Fit and Transform the documents
    train_data = tfidf.fit_transform(documents_list)

    #tfidf_tokens = train_data.get_feature_names()

    #df_tfidfvect = pd.DataFrame(data = train_data.toarray(), index = ['Doc1','Doc2','Doc3', 'Doc4', 'Doc5','Doc6','Doc7' ], columns = tfidf_tokens)

    #print("Train data:")
    #print(train_data[0][0])

    df = pd.DataFrame.sparse.from_spmatrix(train_data, columns=tfidf.get_feature_names_out().tolist())

    for index, row in df.iterrows():
        curr_key = key_list[index]
        folder_index = curr_key.index('short_story')
        curr_key = curr_key[0:folder_index] + 'tf_idf/'+curr_key[folder_index:]
        new_key = curr_key.replace(PREFIX, VECTORIZED_PREFIX)
        json_row = row.to_json()
        s3client.put_object(Bucket=BUCKET_NAME, Key=new_key, Body=json_row)
        #print(row.to_json)
        #print(row['c1'], row['c2'])

    return tfidf

def vectorize_authors_and_get_tf_idf(authors_stories, word_list):
    print("number of author stories:" + str(len(authors_stories)))
    # Initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    authors_stories_transformed = []
    words_shown_up = dict((word, 0) for word in word_list)
    for story in authors_stories:
        new_story = ' '
        for word in story.split(' '):
            if word in word_list:
                new_story = new_story + ' ' + word
                words_shown_up[word] = words_shown_up[word] + 1
        authors_stories_transformed.append(new_story)

    for word in words_shown_up:
        if words_shown_up[word] == 0:
            authors_stories_transformed[-1] = authors_stories_transformed[-1] + ' ' + word


    # Vectorize document using TF-IDF
    tfidf = TfidfVectorizer(lowercase=True,
                            stop_words='english',
                            ngram_range=(1, 1),
                            tokenizer=tokenizer.tokenize)

    # Fit and Transform the documents
    train_data = tfidf.fit_transform(authors_stories_transformed)

    #tfidf_tokens = train_data.get_feature_names()

    #df_tfidfvect = pd.DataFrame(data = train_data.toarray(), index = ['Doc1','Doc2','Doc3', 'Doc4', 'Doc5','Doc6','Doc7' ], columns = tfidf_tokens)

    #print("Train data:")
    #print(train_data[0][0])

    df = pd.DataFrame.sparse.from_spmatrix(train_data, columns=tfidf.get_feature_names_out().tolist())

    for index, row in df.iterrows():
        author = AUTHORS[index]
        print("tf idf size:" + str(len(row)) + " for author:" + author)
        s3client.put_object(Bucket=BUCKET_NAME, Key=VECTORIZED_PREFIX + '/' + author + '/tf_idf_map', Body=row.to_json())

    return tfidf


(author_words_map, document_list, key_list) = generate_document_key_list()
print("finished generate_document_key_list:" + str(author_words_map.keys()))
tfidf = vectorize_documents_and_get_tf_idf(document_list)
print("finished vectorize_documents_and_get_tf_idf ")
tfidf_authors = vectorize_authors_and_get_tf_idf(list(author_words_map.values()), tfidf.get_feature_names_out().tolist())
print("finished vectorize_authors_and_get_tf_idf ")

idf = tfidf_authors.idf_
print("document dictionary size:" + str(len(tfidf.get_feature_names_out().tolist())))
print("author dictionary size:" + str(len(idf.tolist())))

s3client.put_object(Bucket=BUCKET_NAME, Key= 'vectorized_data/entire_dictionary.json', Body=json.dumps(tfidf.get_feature_names_out().tolist()))
s3client.put_object(Bucket=BUCKET_NAME, Key= 'vectorized_data/idf_scores.json', Body=json.dumps(idf.tolist()))
