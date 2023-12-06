from openai import OpenAI
import boto3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csc_matrix
from scipy import sparse
import pandas as pd
import numpy as np
import json
from scipy.sparse.linalg import inv

from scipy.sparse.linalg import lsqr
import sys

from scipy.sparse.linalg import spsolve

#np.set_printoptions(threshold=sys.maxsize)

API_KEY = ''

client = OpenAI(api_key=API_KEY)
AUTHORS = ['C.S. Lewis', 'Ernest Hemingway', 'Jane Austen', 'JK Rowling', 'Shakespeare', 'John Steinbeck',  'Bill Bryson', 'Jhumpa Lahiri', 'JRR Tolkein', 'Amy Tan']

curr_prompt = 'write me a 300 word story in the style of Bill Bryson'

response = client.completions.create(
      model="gpt-3.5-turbo-instruct",
  prompt=curr_prompt,
    max_tokens=3072
)

ACCESS_KEY = ''

SECRET_ACCESS_KEY = ''
BUCKET_NAME = 'ap-math-210'
VECTORIZED_PREFIX = 'vectorized_data'
BYTES_TO_READ = 1024000000

s3client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_ACCESS_KEY)

s3_response_object = s3client.get_object(Bucket=BUCKET_NAME, Key=VECTORIZED_PREFIX + '/entire_dictionary.json')
orig_dictionary = json.loads(s3client.get_object(Bucket=BUCKET_NAME, Key='vectorized_data/entire_dictionary.json')['Body'].read(BYTES_TO_READ))
story = response.choices[0].text
entire_dictionary = []
for i in range(0, len(orig_dictionary)):
  if i % 3 == 0:
    entire_dictionary.append(orig_dictionary[i])


def get_document_tensor(document):
    matrix_count_map = {}
    row_indices = []
    column_indicies = []
    sentence_counter = 0
    data_values = []
    print(document)
    for sentence in document.split('.'):
        words = sentence.split(' ')
        sentence_counter = sentence_counter + 1
        for i in range(0, len(words)):
            for j in range(0, len(words)):
                if i == j or not words[i].isalpha() or not words[j].isalpha():
                    continue

                if (i, j) not in matrix_count_map:
                    try:
                        word_index1 = entire_dictionary.index(words[i].lower())
                        word_index2 = entire_dictionary.index(words[j].lower())
                    except:
                        print("Could not find:" + words[i].lower() + " in dictionary")
                        print("Could not find:" + words[j].lower() + " in dictionary")
                        continue
                    row_indices.append(word_index1)
                    column_indicies.append(word_index2)
                    data_values.append(1.0)
                    matrix_count_map[(word_index1, word_index2)] = (
                    len(row_indices), len(column_indicies), len(data_values))
                else:
                    index_tuple = matrix_count_map[(i, j)]
                    row_index = index_tuple[0]
                    col_index = index_tuple[1]
                    data_index = index_tuple[2]
                    data_values[data_index] = data_values[data_index] + 1.0

        data_values = [item for item in data_values]

        # Populate the sparse matrix using csr_matrix
        sparse_matrix = csc_matrix((data_values, (row_indices, column_indicies)),
                                   shape=(len(entire_dictionary), len(entire_dictionary)))

        return sparse_matrix

def vectorize_matrix(matrix):
    total_entries = matrix.shape
    column_vector = np.zeros(total_entries[0] * total_entries[1])

    coo_matrix = matrix.tocoo()

    for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        column_index = i*total_entries[0] + j
        column_vector[column_index] = v

    return column_vector


author_to_matrix = {}
total_entries = 0
for author in AUTHORS:
    tensor_file_name = "word_cooccurence_matrix.npz"

    object_name = VECTORIZED_PREFIX + '/' + author + '/' + tensor_file_name

    s3_response_object = s3client.get_object(Bucket=BUCKET_NAME, Key=object_name)
    print (s3_response_object['ResponseMetadata'])
    contents = s3_response_object['Body'].read(int(s3_response_object['ResponseMetadata']['HTTPHeaders']['content-length']))

    # Open in "wb" mode to
    # write a new file, or
    # "ab" mode to append
    with open(tensor_file_name, "wb") as binary_file:
        # Write bytes to file
        binary_file.write(contents)

    sparse_matrix = sparse.load_npz(tensor_file_name)

    author_to_matrix[author] = sparse_matrix

    total_entries = sparse_matrix.get_shape()




basis_matrix = []
for author in AUTHORS:

    sparse_matrix = author_to_matrix[author]
    coo_matrix = sparse_matrix.tocoo()
    column_vector = np.zeros(total_entries[0] * total_entries[1])
    for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        column_index = i * total_entries[0] + j
        column_vector[column_index] = v

    if len(basis_matrix) == 0:
        basis_matrix = np.array([column_vector]).T
    else:
        basis_matrix = np.append(basis_matrix, np.array([column_vector]).T, 1)

sparse_basis = sparse.csc_matrix(basis_matrix)
print("got here 1 shape:" + str(sparse_basis.shape))
sparse_basis_transpose = sparse_basis.transpose()
print("got here 2")
first_mat = sparse_basis_transpose * sparse_basis
print("got here 3")
first_mat_inv = inv(first_mat)
print("sparse basis size:" + str(sparse_basis.shape) + " first_mat_inv size:" + str(first_mat_inv.shape) + " sparse_basis_transpose" + str(sparse_basis_transpose.shape))

out_matrix = sparse_basis * first_mat_inv * sparse_basis_transpose
print("got here 5")
document_vector = vectorize_matrix(get_document_tensor(story))
print("got here 6")
projected_vector = out_matrix * document_vector
print("got here 7")
print(document_vector.shape)
print(out_matrix.shape)

out_vector = lsqr(sparse_basis, projected_vector, show=True
                  )

print(out_vector)
print(out_vector[0].shape)
print(out_vector[0])









