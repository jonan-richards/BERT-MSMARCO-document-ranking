import numpy as np
import math

def parse_output_line(line):
    terms = line.split(" ")

    query = int(terms[0])
    document = terms[1]
    score = float(terms[2])

    return query, document, score


class aggregate_scores(object):
    def __init__(self, aggregate):
        self.scores = {}
        self.passage_counts = {}
        self.aggregate = aggregate

    
    def add(self, query, document, score):
        if query not in self.scores:
            self.scores[query] = { document: score }
            self.passage_counts[query] = { document: 1 }
        elif document not in self.scores[query]:
            self.scores[query][document] = score
            self.passage_counts[query][document] = 1
        else:
            self.scores[query][document] = self.aggregate(self.scores[query][document], score)
            self.passage_counts[query][document] += 1


def aggregate_results(input_path, output_path, aggr_method, normalize=False):
    if aggr_method == 'sum':
        aggregate = lambda a, b: a + b
    elif aggr_method == 'max':
        aggregate = lambda a, b: max(a, b)

    aggr_scores = aggregate_scores(aggregate)

    with open(input_path, 'r') as results_file:
        for line in results_file:
            query, document, score = parse_output_line(line)
            aggr_scores.add(query, document, score)
                
    f = open(output_path, "w")
    for query in aggr_scores.scores:
        if normalize:
            for document in aggr_scores.scores[query]:
                aggr_scores.scores[query][document] /= aggr_scores.passage_counts[query][document]

        # rank documents for each query
        ranking = dict(sorted(aggr_scores.scores[query].items(), key=lambda item: item[1], reverse=True))

        rank = 1
        for document in ranking:
            score = aggr_scores.scores[query][document]
            f.write(f'{query} 0 {document} {rank} {score} BERT-{aggr_method}\n')
            rank += 1

    f.close()

# parse single line from QREL file
# output: query ID, document ID and document relevancy
def parse_qrel_line(line):
    terms = line.split(" ")
    
    query = int(terms[0])
    document = terms[2]
    relevancy = int(terms[3])
    
    return query, document, relevancy


# parse single line from results file
# output: query ID, document ID and document ranking
def parse_results_line(line):
    terms = line.split(" ")

    query = int(terms[0])
    document = terms[2]
    rank = int(terms[3])

    return query, document, rank


# lookup for relevancy of document-query pairs
# get returns zero if pair is not in lookup
class relevancy_lookup(object):
    def __init__(self):
        self.relevancies = {}
    
    def add(self, query, document, relevancy):
        if query not in self.relevancies:
            self.relevancies[query] = {}

        self.relevancies[query][document] = relevancy
        
    def get(self, query, document):
        if document not in self.relevancies[query]:
            return 0

        return self.relevancies[query][document]


def get_ranked_labels(rel_lookup, query, doc_rank_list): 
    result = np.zeros(len(doc_rank_list), dtype=int)
    for x in doc_rank_list:
        result[x[1]-1] = rel_lookup.get(query, x[0])
    return result

def get_ranked_labels(rel_lookup, query, doc_rank_list): 
    result = np.zeros(len(doc_rank_list), dtype=int)
    for x in doc_rank_list:
        result[x[1]-1] = rel_lookup.get(query, x[0])
    return result

def process_files(qrel_path, results_path):
    relevancies = relevancy_lookup()
    with open(qrel_path, 'r') as qrel_file:
        for line in qrel_file:
            query, document, relevancy = parse_qrel_line(line)
            relevancies.add(query, document, relevancy)
    with open(results_path, 'r') as results_file:
        current_query, document, rank = parse_results_line(next(results_file))    
        doc_rank_list = [(document, rank)]
        for line in results_file:
            query, document, rank = parse_results_line(line)
            if query != current_query:
                yield get_ranked_labels(relevancies, current_query, doc_rank_list)
                current_query = query
                doc_rank_list = [(document, rank)]
            else:
                doc_rank_list.append((document, rank))
        yield get_ranked_labels(relevancies, current_query, doc_rank_list)

def precision(query_relevancy_labels, k):
    return sum(query_relevancy_labels[:k]) / k


def recall(query_relevancy_labels, k):
    if (sum(query_relevancy_labels) == 0):
        return 0.0
    
    return sum(query_relevancy_labels[:k]) / sum(query_relevancy_labels)


def F_score(query_relevancy_labels, k):
    p = precision(query_relevancy_labels, k)
    r = recall(query_relevancy_labels, k) 
    if (p + r == 0.0):
        return 0

    return (2 * p * r) / (p + r)


def DCG(query_relevancy_labels, k):
    # Use log with base 2
    dcg = 0
    for i in range(0, min(k, len(query_relevancy_labels))):
        dcg += query_relevancy_labels[i] / math.log2(2 + i)

    return dcg


def NDCG(query_relevancy_labels, k):
    nr_relevant = min(sum(query_relevancy_labels), k)
    IDCG = DCG([1] * nr_relevant, k)
    if IDCG == 0.0:
        return 0

    return DCG(query_relevancy_labels, k) / IDCG


def AP(query_relevancy_labels):
    nr_queries = len(query_relevancy_labels)
    nr_relevant = min(sum(query_relevancy_labels), nr_queries)
    
    if nr_relevant == 0.0:
        return 0

    avg_precision = 0
    for k in range(0, nr_queries):
        avg_precision += query_relevancy_labels[k] * precision(query_relevancy_labels, k+1) / nr_relevant

    return avg_precision


def RR(query_relevancy_labels):
    i = np.where(query_relevancy_labels == 1)[0]
    if len(i) == 0:
        return 0

    return 1 / (1 + i[0])

def evaluate(qrel_path, results_path):
    results_per_query = {
        'precision@1': [],
        'precision@5': [],
        'precision@10': [],
        'precision@25': [],
        'recall@1': [],
        'recall@5': [],
        'recall@10': [],
        'recall@25': [],
        'F-score@1': [],
        'F-score@5': [],
        'F-score@10': [],
        'F-score@25': [],
        'DCG@1': [],
        'DCG@5': [],
        'DCG@10': [],
        'DCG@25': [],
        'NDCG@1': [],
        'NDCG@5': [],
        'NDCG@10': [],
        'NDCG@25': [],
        'MAP': [],
        'MRR': [],
    }
    for labels in process_files(qrel_path, results_path):
        results_per_query['precision@1'].append(precision(labels, 1))
        results_per_query['precision@5'].append(precision(labels, 5))
        results_per_query['precision@10'].append(precision(labels, 10))
        results_per_query['precision@25'].append(precision(labels, 25))
        results_per_query['recall@1'].append(recall(labels, 1))
        results_per_query['recall@5'].append(recall(labels, 5))
        results_per_query['recall@10'].append(recall(labels, 10))
        results_per_query['recall@25'].append(recall(labels, 25))
        results_per_query['F-score@1'].append(F_score(labels, 1))
        results_per_query['F-score@5'].append(F_score(labels, 5))
        results_per_query['F-score@10'].append(F_score(labels, 10))
        results_per_query['F-score@25'].append(F_score(labels, 25))
        results_per_query['DCG@1'].append(DCG(labels, 1))
        results_per_query['DCG@5'].append(DCG(labels, 5))
        results_per_query['DCG@10'].append(DCG(labels, 10))
        results_per_query['DCG@25'].append(DCG(labels, 25))
        results_per_query['NDCG@1'].append(NDCG(labels, 1))
        results_per_query['NDCG@5'].append(NDCG(labels, 5))
        results_per_query['NDCG@10'].append(NDCG(labels, 10))
        results_per_query['NDCG@25'].append(NDCG(labels, 25))
        results_per_query['MAP'].append(AP(labels))
        results_per_query['MRR'].append(RR(labels))
    
    results = {}
    for key, values in results_per_query.items():
        results[key] = np.mean(values)
    return results