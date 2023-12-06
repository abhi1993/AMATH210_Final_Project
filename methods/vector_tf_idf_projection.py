from openai import OpenAI
import boto3
import json
import sys
import numpy as np
import re
import copy

ACCESS_KEY = ''

SECRET_ACCESS_KEY = ''
BUCKET_NAME = 'ap-math-210'
VECTORIZED_PREFIX = 'vectorized_data'
BYTES_TO_READ = 1024000000

AUTHORS = ['C.S. Lewis', 'Ernest Hemingway', 'Jane Austen', 'JK Rowling', 'Shakespeare', 'John Steinbeck',  'Bill Bryson', 'Jhumpa Lahiri', 'JRR Tolkein',  'Amy Tan']
s3client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_ACCESS_KEY)

entire_dictionary = json.loads(s3client.get_object(Bucket=BUCKET_NAME, Key='vectorized_data/entire_dictionary.json')['Body'].read(BYTES_TO_READ))
s3_response_object = s3client.get_object(Bucket=BUCKET_NAME, Key=VECTORIZED_PREFIX + '/idf_scores.json')
idf_scores = json.loads(s3_response_object['Body'].read(BYTES_TO_READ))

API_KEY = ''

client = OpenAI(api_key=API_KEY)



def get_matrix(s3client):
    ret_matrix = []
    for author in AUTHORS:
        key = VECTORIZED_PREFIX + '/' + author + '/tf_idf_map'
        print("getting key:" + key)
        probability_vector = s3client.get_object(Bucket=BUCKET_NAME, Key=key)
        word_scores = json.loads(probability_vector['Body'].read().decode('utf-8'))
        print(len(word_scores.values()))
        if len(ret_matrix) == 0:
            ret_matrix = np.array([list(word_scores.values())]).T
        else:
            n_column = np.array([list(word_scores.values())]).T
            ret_matrix = np.append(ret_matrix, n_column, 1)
    return ret_matrix

def get_document_tf_idf_score(document, idf_scores, entire_dictionary):

    index_to_count = {}
    total_words = 0

    for word in document.split(' '):
        word = re.sub(r'[^A-Za-z0-9 ]+', '', word)
        word = word.lower()
        if word in entire_dictionary:
            word_index = entire_dictionary.index(word)

            if not word_index in index_to_count:
                index_to_count[word_index] = 1
            else:
                index_to_count[word_index] = index_to_count[word_index] + 1

        total_words = total_words + 1

    ret_arr = np.zeros(len(idf_scores))

    for index in range(0, len(ret_arr)):
        if index in index_to_count:
            word_count = index_to_count[index]/total_words
            ret_arr[index] = word_count * idf_scores[index]
    return ret_arr


def classify_prompts(prompts, out_matrix):
    number_completely_correct = 0
    number_partially_correct = 0
    correct_answers = []
    for prompt in prompts:
        response = client.completions.create(
              model="gpt-3.5-turbo-instruct",
          prompt=prompt[0],
            max_tokens=3072
        )
        story = response.choices[0].text
        document_tf_idf_score = get_document_tf_idf_score(story, idf_scores, entire_dictionary)

        projected_vector = np.matmul(out_matrix, document_tf_idf_score)

        np.set_printoptions(threshold = sys.maxsize)

        components = np.linalg.lstsq(basis_matrix, projected_vector)
        component_arr = components[0]
        index = 0
        total_components = 0
        for component in component_arr:
            if component > 0:
                total_components = total_components + component
            index = index + 1

        max_indices = np.argpartition(component_arr, -2)[-2:]
        print("For prompt:" + prompt[0])
        influence1 = component_arr[max_indices[0]]/total_components
        influence2 = component_arr[max_indices[1]] / total_components
        print("Top contributors:" + AUTHORS[max_indices[0]] + " and " + AUTHORS[max_indices[1]] + " with influence:" +
              str(influence1) + " and " + str(influence2))
        print("Expected authors: " + str(prompt[1]))
        print('\n\n\n')

        if len(prompt[1]) == 1 and prompt[1][0] == AUTHORS[max_indices[1]]:
            number_completely_correct = number_completely_correct + 1
            print('COMPLETELY CORRECT')
            correct_answers.append('COMPLETELY CORRECT')
        elif set(prompt[1]) == set([AUTHORS[max_indices[0]], AUTHORS[max_indices[1]]]):
            number_completely_correct = number_completely_correct + 1
            print('COMPLETELY CORRECT')
            correct_answers.append('COMPLETELY CORRECT')
        elif len(set(prompt[1]).intersection(set([AUTHORS[max_indices[0]], AUTHORS[max_indices[1]]]))) > 0:
            number_partially_correct = number_partially_correct + 1
            print('PARTIALLY CORRECT')
            correct_answers.append('PARTIALLY CORRECT')
        else:
            correct_answers.append('WRONG')

    print("Number of completely correct predictions:" + str(number_completely_correct))
    print("Number of partially correct predictions:" + str(number_partially_correct))
    print("Correct predictions list:" + str(correct_answers))


basis_matrix = get_matrix(s3client)
out_mat = np.matmul(np.matmul(basis_matrix, np.linalg.inv(np.matmul(basis_matrix.T, basis_matrix))), basis_matrix.T)



prompt_matrix = [
('write me a 300 word short story using the iceberg theory of writing', ['Ernest Hemingway']),
('write me a 300 word story in the style of C.S. Lewis', ['C.S. Lewis']),
('write me a 300 word story in the style of Ernest Hemingway',['Ernest Hemingway']),
('write me a 300 word story in the style of Jane Austen', ['Jane Austen']),
('write me a 300 word story in the style of JK Rowling', ['JK Rowling']),
('write me a 300 word story in the style of Shakespeare', ['Shakespeare']),
('write me a 300 word story in the style of John Steinbeck', ['John Steinbeck']),
('write me a 300 word story in the style of Bill Bryson', ['Bill Bryson']),
('write me a 300 word story in the style of Jhumpa Lahiri', ['Jhumpa Lahiri']),
('write me a 300 word story in the style of JRR Tolkein', ['JRR Tolkein']),
('write me a 300 word story in the style of Amy Tan', ['Amy Tan']),
('write me a 300 word short story about a California farmer', ['John Steinbeck']),
('write me a 300 word short story about a love story set in England', ['Shakespeare']),
('write me a 300 word short story about a love story set in Victorian England', ['Jane Austen']),
('write me a 300 word short story about an immigrant family', ['Jhumpa Lahiri', 'Amy Tan']),
('write me a 300 word short story set in a fantasy world with many animals', ['C.S. Lewis', 'JRR Tolkein']),
('write me a 300 word short story about war', ['Ernest Hemingway', 'JRR Tolkein']),
('write me a 300 word short story about violence and death', ['Ernest Hemingway', 'JRR Tolkein']),
('write me a 300 word comedic story about a long hike', ['Bill Bryson']),
('write me a 300 word comedic story with biblical allegories', ['John Steinbeck', 'CS Lewis']),
('write me a 300 word story about a Chinese immigrant family', ['Amy Tan'])]

for i in range(0, len(AUTHORS)):
    for j in range(i+1, len(AUTHORS)):
        prompt = 'write me a 300 word story in the style of ' + AUTHORS[i] + ' and '+ AUTHORS[j]
        prompt_matrix.append((prompt, [AUTHORS[i], AUTHORS[j]]))


classify_prompts(prompt_matrix, out_mat)