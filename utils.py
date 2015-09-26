import csv
from collections import Counter
from IPython import embed
from re import sub
from random import randint
from gensim import corpora, models, similarities

def dot(phrase1, phrase2):
    """
    Defines the similarity between a sample
    and a list of words
    """
    score = 0
    for word in phrase2:
        if word in phrase1:
            score += phrase1[word] * phrase2[word]
    return score

def create_sample_vector(sample):
    """
    Creates a vector out of a samples
    """
    word_dict = Counter()
    for el in sample[1].lower().split():
        word_dict[el] += 1
    for el in sample[2].lower().split():
        word_dict[el] += 1
    for el in sample[6].lower().split():
        word_dict[el] += 1
    return word_dict

def create_vector(sample):
    word_dict = Counter()
    for word in sample:
        word_dict += 1
    return word_dict


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

def split_csv(name, parts):
    length = 0
    with open(name) as f:
        csvreader = csv.reader(f)
        csvreader.next()
        while True:
            length += 1
            try:
                line = csvreader.next()
            except:
                break
    print(length)

    with open(name) as f:
        csvreader = csv.reader(f)
        header = str(csvreader.next())
        header = header[1:len(header)-1]
        name = name.split('.')
        for i in range(parts):
            new_name = [name[0]+`i`, name[1]]
            with open('.'.join(new_name),'w') as g:
                g.write(header)
                for j in range(length//parts):
                    line = str(csvreader.next())
                    g.write(line[1:len(line)-1])
                g.flush()
        line = csvreader.next()
        if line:
            new_name = [name[0]+`i+1`, name[1]]
            with open('.'.join(new_name), 'w') as g:
                g.write(header)
                while True:
                    line = str(csvreader.next())
                    g.write(line[1:len(line)-1])
                    try:
                        line = csvreader.next()
                    except:
                        break
                g.flush()
#split_csv('output.csv', 10)



# def cluster(k, data):
#     vectored_data = []
#     for datum in data:
#         vectored_data.append(create_input_vector(datum))
#     centers

# while True:
#     entered = raw_input("Enter your question: ")
#     entered = create_input_vector(entered.lower().split())
#     best_answer = ""
#     best_score = 0
#     best_match = ""
#     for i in range(len(vectors)):
#         answer = answers[i]
#         question = vectors[i]
#         score = dot(question, entered)
#         if score > best_score:
#             best_answer = answer
#             best_score = score
#             best_match = questions[i]
#     print(best_answer + "\n")

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
def best_answer(question, answers):
    new_vec = dictionary.doc2bow(question.split())
    sims = list(index[tfidf[new_vec]])
    return answers[sims.index(max(sims))]




