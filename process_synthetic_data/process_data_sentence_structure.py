import boto3
import nltk
from nltk.corpus import stopwords


BUCKET = 'ap-math-210'
RAW_FOLDER = 'raw_data'
DELIMITER = '/'
BYTES_TO_READ = 4096

access_key = ''
secret_access_key = ''

# Let's use Amazon S3
s3 = boto3.client('s3', aws_access_key_id=access_key,
    aws_secret_access_key=secret_access_key)

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
english_stopwords = stopwords.words('english')


response = s3.list_objects_v2(Bucket=BUCKET, Prefix=RAW_FOLDER)

counter = 0


while 'NextContinuationToken' in response:
    NEXT_TOKEN = response['NextContinuationToken']
    print("continuation token:" + str(NEXT_TOKEN))
    for paragraph in response['Contents']:
        key = paragraph['Key']

        if key[-1] != '/':
            print("key is:" + key)
            story = str(s3.get_object(Bucket=BUCKET, Key=key)['Body'].read(BYTES_TO_READ))
            story = story.replace('\\n', ' ').replace('"', ' ')
            print(story)
            final_paragraph = ''
            for sentence in story.split('.'):
                tagged_sentence = nltk.tag.pos_tag(sentence.split())
                print(sentence)
                print(tagged_sentence)
                edited_sentence = [word.lower() for word, tag in tagged_sentence[1:] if word.lower() not in english_stopwords and word.lower().isalpha()]
                final_sentence = ' '.join(edited_sentence) + '.'
                final_paragraph = final_paragraph + final_sentence

            print(final_paragraph)
            new_key = key.replace('raw_data', 'processed_data_sentence_structure')
            s3.put_object(Bucket=BUCKET, Key=new_key
                          , Body=final_paragraph)

    response = s3.list_objects_v2(Bucket=BUCKET, ContinuationToken=NEXT_TOKEN, Prefix=RAW_FOLDER)







