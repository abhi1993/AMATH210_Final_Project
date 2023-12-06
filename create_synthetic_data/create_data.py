from openai import OpenAI
import boto3


API_KEY = ''
AUTHORS = [ 'Jhumpa Lahiri', 'Bill Bryson', 'Amy Tan']

PROMPT = "Write a 150 word short story in the style of "

access_key = ''
secret_access_key = ''

# Let's use Amazon S3
s3 = boto3.client('s3', aws_access_key_id=access_key,
    aws_secret_access_key=secret_access_key)
bucket_name = 'ap-math-210'
NUM_STORIES = 100
client = OpenAI(api_key=API_KEY)

for author in AUTHORS:

    for num in range(0, NUM_STORIES):
        curr_prompt = PROMPT + author
        response = client.completions.create(
              model="gpt-3.5-turbo-instruct",
          prompt=curr_prompt,
            max_tokens=1024
        )

        #print(response.choices[0].text)
        key = 'raw_data/' + author + '/short_story' + str(num)
        s3.put_object(Bucket=bucket_name, Key=key,  Body=response.choices[0].text)
