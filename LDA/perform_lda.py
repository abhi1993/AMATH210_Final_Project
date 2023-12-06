import boto3
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


ACCESS_KEY = ''

SECRET_ACCESS_KEY = ''
BUCKET_NAME = 'ap-math-210'
PREFIX = 'processed_data'

s3client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_ACCESS_KEY)

response = s3client.list_objects_v2(Bucket = BUCKET_NAME, Prefix =PREFIX)
documents_list = []
key_list = []

for object in response['Contents']:

    #if 'Hemingway' in object['Key'] or 'Shakespeare' in object['Key'] or 'Steinbeck' in object['Key']:
    s3_response_object = s3client.get_object(Bucket=BUCKET_NAME, Key=object['Key'])
    object_content = s3_response_object['Body'].read()
    documents_list.append(object_content)
    key_list.append(object['Key'])

# Initialize regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# Vectorize document using TF-IDF
tfidf = TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer=tokenizer.tokenize)

# Fit and Transform the documents
train_data = tfidf.fit_transform(documents_list)

# Define the number of topics or components
num_components = 7

# Create LDA object
model=LatentDirichletAllocation(n_components=num_components)

# Fit and Transform SVD model on data
lda_matrix = model.fit_transform(train_data)

# Get Components
lda_components=model.components_


# Print the topics with their terms
terms = tfidf.get_feature_names_out()

for index, component in enumerate(lda_components):
    zipped = zip(terms, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:7]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ", top_terms_list)


doc_topic = model.transform(train_data)

for n in range(doc_topic.shape[0]):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}, topic array: {}, key:{}\n".format(n, topic_most_pr, doc_topic[n], key_list[n]))
    #print(documents_list[n])