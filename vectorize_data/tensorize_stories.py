import boto3
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csr_matrix
from scipy import sparse
import pandas as pd
import numpy as np
import json

ACCESS_KEY = ''

SECRET_ACCESS_KEY = ''
BUCKET_NAME = 'ap-math-210'
PREFIX = 'processed_data_sentence_structure'
VECTORIZED_PREFIX = 'vectorized_data'

AUTHORS = [ 'C.S. Lewis', 'Ernest Hemingway', 'Jane Austen', 'Jhumpa Lahiri', 'JK Rowling', 'Shakespeare', 'John Steinbeck',  'Bill Bryson', 'JRR Tolkein','Amy Tan']
BYTES_TO_READ = 1024000000

s3client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_ACCESS_KEY)

s3_response_object = s3client.get_object(Bucket=BUCKET_NAME, Key=VECTORIZED_PREFIX + '/entire_dictionary.json')
orig_dictionary = json.loads(
    s3client.get_object(Bucket=BUCKET_NAME, Key='vectorized_data/entire_dictionary.json')['Body'].read(BYTES_TO_READ))

entire_dictionary = []
for i in range(0, len(orig_dictionary)):
  if i % 3 == 0:
    entire_dictionary.append(orig_dictionary[i])

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def generate_document_key_list():
    s3client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_ACCESS_KEY)

    response = s3client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    documents_list = []
    key_list = []
    author_words_map = dict((author, ' ') for author in AUTHORS)
    iteration = 0
    while 'NextContinuationToken' in response or iteration == 0:

        for object in response['Contents']:
            curr_key = object['Key']
            folder_indices = find(curr_key, '/')
            author = curr_key[folder_indices[-2] + 1:folder_indices[-1]]
            # if 'Hemingway' in object['Key'] or 'Shakespeare' in object['Key'] or 'Steinbeck' in object['Key']:
            s3_response_object = s3client.get_object(Bucket=BUCKET_NAME, Key=curr_key)
            object_content = s3_response_object['Body'].read()
            author_words_map[author] = author_words_map[author] + ' ' + str(object_content)
            documents_list.append(object_content)
            print("Looking at key:" + curr_key)
            key_list.append(curr_key)
        iteration = iteration + 1

        if 'NextContinuationToken' in response:
            NEXT_TOKEN = response['NextContinuationToken']
            response = s3client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX, ContinuationToken=NEXT_TOKEN)

    return (author_words_map, documents_list, key_list)

def create_author_tensors(author_words_map):
    author_tensor_map = {}
    for author in author_words_map:
        author_words = author_words_map[author]
        dict_len = len(entire_dictionary)
        author_word_set = set()
        author_tensor_map[author] = np.zeros((dict_len, dict_len))
        row_indices = []
        column_indicies = []
        data_values = []
        matrix_count_map = {}
        sentence_counter = 0
        for sentence in author_words.split('.'):
            words = sentence.split(' ')
            sentence_counter = sentence_counter + 1
            for i in range(0, len(words)):
                for j in range(0, len(words)):
                    if i == j or not words[i].isalpha() or not words[j].isalpha():
                        continue

                    if (i, j) not in matrix_count_map:
                        try:
                            word_index1 = entire_dictionary.index(words[i])
                            word_index2 = entire_dictionary.index(words[j])
                        except:
                            continue
                        row_indices.append(word_index1)
                        column_indicies.append(word_index2)
                        data_values.append(1.0)
                        matrix_count_map[(word_index1, word_index2)] = (len(row_indices), len(column_indicies), len(data_values))
                    else:
                        index_tuple = matrix_count_map[(i, j)]
                        row_index = index_tuple[0]
                        col_index = index_tuple[1]
                        data_index = index_tuple[2]
                        data_values[data_index] = data_values[data_index] + 1.0

            data_values = [item for item in data_values]

        # Populate the sparse matrix using csr_matrix
        sparse_matrix = csr_matrix((data_values, (row_indices, column_indicies)), shape=(len(entire_dictionary), len(entire_dictionary)))



        #print("Now building tensor for author: " + author + " number of unique words:" + str(len(author_word_set)))
        #curr_tensor = []
        #for i in range(0, len(author_word_set)):
        #    for j in range(0, len(author_word_set)):
        #        if i == j:
        #            continue
        #        curr_tensor.append((i, j))

        #    print("Looking at row: " + str(i))
        author_tensor_map[author] = sparse_matrix



    return author_tensor_map


(author_words_map, documents_list, key_list) = generate_document_key_list()

print(author_words_map)

author_tensor_map = create_author_tensors(author_words_map)


for author in author_tensor_map:
    csr_tensor = author_tensor_map[author]
    tensor_file_name = "word_cooccurence_matrix"
    sparse.save_npz(tensor_file_name, csr_tensor)
    object_name =  VECTORIZED_PREFIX + '/' + author + '/' + tensor_file_name + ".npz"
    response = s3client.upload_file(tensor_file_name + '.npz', BUCKET_NAME, object_name)
    print(response)


print(author_tensor_map['Ernest Hemingway'])



