from openai import OpenAI
import boto3
import json
import sys
import numpy as np



API_KEY = ''

client = OpenAI(api_key=API_KEY)
AUTHORS = ['C.S. Lewis', 'Ernest Hemingway', 'Jane Austen', 'JK Rowling', 'Shakespeare', 'John Steinbeck',  'Bill Bryson', 'Jhumpa Lahiri', 'JRR Tolkein','Amy Tan']

curr_prompt = 'write me a 300 word story about an Indian immigrant family'

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
entire_dictionary = json.loads(s3client.get_object(Bucket=BUCKET_NAME, Key='vectorized_data/entire_dictionary.json')['Body'].read(BYTES_TO_READ))
story = response.choices[0].text

def get_matrix(s3client):
    ret_matrix = []
    for author in AUTHORS:
        key = VECTORIZED_PREFIX + '/' + author + '/probability_map'
        print("getting key:" + key)
        probability_vector = s3client.get_object(Bucket=BUCKET_NAME, Key=key)
        word_dict = json.loads(probability_vector['Body'].read().decode('utf-8'))
        print("matrix type:" + str(type(ret_matrix)))
        if len(ret_matrix) == 0:
            ret_matrix = np.array([list(word_dict.values())]).T
        else:
            #print(ret_matrix.shape)
            #print([list(word_dict.values())].shape)
            n_column = np.array([list(word_dict.values())]).T
            ret_matrix = np.append(ret_matrix, n_column, 1)
    return ret_matrix

def create_document_vector(story, curr_word_count_map):
    total_words = 0
    for word in story.split(' '):
        lower_word = word.lower()
        final_word = ''.join([i for i in lower_word if i.isalpha()])
        if final_word in entire_dictionary:
            if final_word not in curr_word_count_map:
                curr_word_count_map[final_word] = 1
            else:
                curr_word_count_map[final_word] = curr_word_count_map[final_word] + 1

            total_words = total_words + 1

    for key in curr_word_count_map:
        curr_word_count_map[key] = curr_word_count_map[key] / total_words

    output_vec = np.array(list(curr_word_count_map.values()))
    return output_vec


def classify_prompts(prompts, out_matrix):
    number_completely_correct = 0
    number_partially_correct = 0
    for prompt in prompts:
        response = client.completions.create(
              model="gpt-3.5-turbo-instruct",
          prompt=prompt[0],
            max_tokens=3072
        )
        story = response.choices[0].text
        curr_word_count_map = dict((el, 0) for el in entire_dictionary)
        document_tf_idf_score = create_document_vector(story, curr_word_count_map )

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

        if len(prompt[1]) == 1 and prompt[1][0] == AUTHORS[max_indices[1]]:
            number_completely_correct = number_completely_correct + 1
            print('COMPLETELY CORRECT')
        elif set(prompt[1]) == set([AUTHORS[max_indices[0]], AUTHORS[max_indices[1]]]):
            number_completely_correct = number_completely_correct + 1
            print('COMPLETELY CORRECT')
        elif len(set(prompt[1]).intersection(set([AUTHORS[max_indices[0]], AUTHORS[max_indices[1]]]))) > 0:
            number_partially_correct = number_partially_correct + 1
            print('PARTIALLY CORRECT')

        print('\n\n\n')
    print("Number of completely correct predictions:" + str(number_completely_correct))
    print("Number of partially correct predictions:" + str(number_partially_correct))


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








