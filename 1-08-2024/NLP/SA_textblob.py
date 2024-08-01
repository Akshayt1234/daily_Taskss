#textblob library
#create a sample textimport textblob over
#let 
from textblob import TextBlob

texts=[
    "I Love NLP ! It's works great and I'm very satisfied ",
    "This is my first experience on doing sentiment analysis , I am little bit disappointed",
    "The NLP sentiment analysis is quiet interesting  it is neither good or bad ",
]

#create function to do the sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    #-1.0 - 1.0
    polarity = analysis.sentiment.polarity
    if polarity>0:
        sentiment="positieve"
    elif polarity<0:
        sentiment="Negative"
    else:
        sentiment="Neutral" 
    return sentiment

   
for text in texts:
    sentiment= analyze_sentiment(text)
    print(f"Text : {text}")
    print(f"sentiment: {sentiment}\n")           
    
    