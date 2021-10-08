"""
Input the statement you want to be verified.
Then get and analyze relevant content from Wikipedia.org.
Compare with the input_text.
"""

# Enter the statement you want to be verified here.
# input_text = 'The US population is 300 million.'
# input_text = 'Micheal Jordan is the best NBA player.'
input_text = 'Squirrel cage motors are induction motors.'

import requests
from bs4 import BeautifulSoup
import stanza
from sentence_transformers import SentenceTransformer, util
from stanza.server import CoreNLPClient

en_nlp = stanza.Pipeline('en', verbose=False)


def get_wiki(input_text: str) -> str:
    """
    Search input_text from Wikipedia. Returns the relevant article.
    """
    response = requests.get(f'https://en.wikipedia.org/w/index.php?search={input_text}&title=Special%3ASearch&fulltext=1')
    bs = BeautifulSoup(response.text, "html.parser")
    links_list = bs.find('ul', class_='mw-search-results')
    links = []
    for a_tag in links_list.find_all('a'):
        links.append('https://en.wikipedia.org' + a_tag.get('href'))

    # For now, only use the first link which is the most relevant.
    response = requests.get(links[0])
    bs = BeautifulSoup(response.text, "html.parser")
    content = ''
    for p in bs.find_all('p'):
        content = content + p.text + ' '
    return content


def most_similar(input_text: str, wiki_content: str) -> (str, int):
    """
    Find the sentence with highest similarity score.
    """
    # Let StanfordNLP analyze it.
    en_doc = en_nlp(wiki_content)
    # Parse into sentences, and put them into a list.
    sentences = []
    sentences_sentiment = []
    for sent in en_doc.sentences:
        sentences.append(sent.text)
        sentences_sentiment.append(sent.sentiment)

    print('Calculating similarities...')
    # Use Sentence-Transformer to find the similarities.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Compute embedding for both lists.
    embeddings1 = model.encode([input_text], convert_to_tensor=True)
    embeddings2 = model.encode(sentences, convert_to_tensor=True)
    # Compute cosine-similarities.
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    # Output the sentence with highest similarity score.
    sentence_scores = list(cosine_scores[0])
    max_index = sentence_scores.index(max(sentence_scores))
    most_similar_sent = sentences[max_index]
    most_similar_sent_sentiment = sentences_sentiment[max_index]
    return most_similar_sent, most_similar_sent_sentiment


def get_local_stanfordNLP(text: str, annotators: str):
    """
    Not recommend, please use function get_remote_stanfordNLP.
    Get stanfordnlp result from local device.
    text should be a string, it can include several sentences.
    annotators should be a string, such as 'tokenize,ssplit,pos,lemma,ner,parse,depparse,coref'.
    For the output type and how to obtain specific data in it, please see the official guidance from stanfordNLP or see the function get_openidtriple.
    """
    with CoreNLPClient(
            properties={'annotators': annotators, 'coref.algorithm': 'statistical'},
            timeout=30000,
            memory='6G') as client:
        ann = client.annotate(text)
    return ann


def get_remote_stanfordNLP(input: str, annotators: str) -> str:
    """
    Get stanfordnlp result from remote devices, this is faster than getting result from local device.
    input should be a string, it can include several sentences.
    annotators should be a string, such as 'tokenize,ssplit,pos,lemma,ner,parse,depparse,coref'.
    For the output type and how to obtain specific data in it, please see stanfordNLP official guidance
    """
    return requests.post('http://66.76.242.198:9888/?properties={"annotators":' + annotators + ',"outputFormat":"json"}', data=input).text


def split_doc(document: str) -> list:
    """
    Split document into separate sentences.
    document should be a string, it can include several sentences.
    Output is a list such as ['Chris Manning is a nice person.', 'Chris wrote a simple sentence.']
    """
    doc = en_nlp(document)
    sent_list = [sent.text for sent in doc.sentences]
    return sent_list


def get_openidtriple(text: str) -> list:
    """
    Get subject, relation, object triple-pairs of a sentence from local device.
    text should be a string, it can include several sentences.
    The output is a list such as [['Chris Manning', 'is', 'nice person'], ['Chris Manning', 'is', 'person'], ['Manning', 'is', 'nice']]
    """
    ann = get_local_stanfordNLP(text)
    all_pairs = []
    # get the openidTriple
    for sentence in ann.sentence:
        for triple in sentence.openieTriple:
            all_pairs.append([triple.subject, triple.relation, triple.object])
    return all_pairs


def mapping(statement: str, wikitext: str):
    """
    Check if data from wiki has the same triple-pair with input
    statement should be a str with one sentence, multiple sentences will be added if needed.
    wikitext can be a str with multiple sentences
    """
    if statement is None:
        return None
    if len(split_doc(statement)) == 1:
        sta_triple = get_openidtriple(statement)
        wiki_triple = get_openidtriple(wikitext)
        for wt in wiki_triple:
            for st in sta_triple:
                if st == wt:
                    return True
        return False
    else:
        pass


if __name__ == '__main__':
    # Please comment out the following line after the first run.
    stanza.download('en')

    # en_nlp = stanza.Pipeline('en')
    # Get the sentiment of input_text.
    en_doc = en_nlp(input_text)
    input_sentiment = en_doc.sentences[0].sentiment

    # Get content from Wikipedia
    wiki_content = get_wiki(input_text).replace('\n', ' ')

    # Get the most similar sentence.
    similar_sent, similar_sent_sentiment = most_similar(input_text, wiki_content)
    print(f'The input statement is\n\n{input_text}.\n\n\n')
    print(f'The most similar sentence is\n\n{similar_sent}.\n\n')

    if input_sentiment == similar_sent_sentiment:
        print('The sentiment is same.')
    else:
        print('The sentiment is different.')


