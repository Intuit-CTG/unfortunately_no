from flask import Flask, render_template, request
import csv
from collections import Counter
from gensim import corpora, models, similarities


def parse_samples(num_samples):
    """
    Parses the data from the zip file
    """
    answers = []
    vectors = []
    questions = []
    with open("output.csv") as f:
        csvreader = csv.reader(f)
        csvreader.next()
        for __ in range(num_samples):
            cur = csvreader.next()
            answers.append(cur[8])
            vectors.append(create_sample_vector(cur))
            answers.append(cur[8])
            questions.append(cur[2])
    return answers, vectors, questions

answers, vectors, questions = parse_samples(5000)

stopList = set("for a of the in to and but my".split())
texts = [[word for word in question.lower().split() if word not in stopList]
            for question in questions]

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
            for text in texts]

dictionary = corpora.Dictionary(texts)
dictionary.save('temp.dict')

new_vec = dictionary.doc2bow('taxes are hard'.split())
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
index = similarities.SparseMatrixSimilarity(tfidf[corpus],\
             num_features=len(dictionary))
def best_answer(question, answers, index, tfidf):
    new_vec = dictionary.doc2bow(question.split())
    sims = list(index[tfidf[new_vec]])
    return answers[sims.index(max(sims))]

app = Flask(__name__)

@app.route('/')
def aboutlandingpage():
    return render_template('message.html')


@app.route('/signup', methods = ['GET', 'POST'])
def getAd():
    if request.method == 'POST':
        selection1 = request.form['Relationship']
        selection2 = request.form['Family']
        selection3 = request.form['Housing']
        selection4 = request.form['postBox']


    #call selection on function that back-end people write and return relevant information

    return render_template("news_feed.html", selection= best_answer(selection1 + ' ' +\
             selection2 + ' ' + selection3 + ' ' + selection4, answers,index, tfidf), final = selection4)






if __name__ == "__main__":
    app.run()
