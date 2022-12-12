import re
import unidecode
import torch
from tqdm.notebook import tqdm

# function to get the IDs of the previous queries of a query in a session
# from https://github.com/Tomjg14/Master_Thesis_MSMARCO_Passage_Reranking_BERT
def get_lower_ids(session_df, query_id):
    session_id = int(query_id.split('_')[0])
    current_id = int(query_id.split('_')[1])
    all_ids = [int(x.split('_')[1]) for x in session_df['query_id'].tolist()]
    lower_ids = [x for x in all_ids if x < current_id]
    lower_ids = [str(session_id) + '_' + str(x) for x in lower_ids]
    return lower_ids

# function that strips all non-alphanumeric characters
# from https://github.com/Tomjg14/Master_Thesis_MSMARCO_Passage_Reranking_BERT
def remove_non_alphanumeric(text, keep_periods=False):
    text = unidecode.unidecode(str(text))
    text = re.sub(r'[^a-zA-Z0-9.]' if keep_periods else r'[^a-zA-Z0-9]', ' ', text)
    return text

# function that returns a list of segment ids based on indexed tokens (BERT)
# from https://github.com/Tomjg14/Master_Thesis_MSMARCO_Passage_Reranking_BERT
def get_segment_ids_from_index_tokens(indexed_tokens):
    segment_ids = []
    sep = False
    for i, token in enumerate(indexed_tokens):
        if token == 102:
            sep = True
        if sep:
            segment_ids.append(1)
        else:
            segment_ids.append(0)
    return segment_ids

# from https://github.com/Tomjg14/Master_Thesis_MSMARCO_Passage_Reranking_BERT
def run_bert(model, data):
    activations = []
    for i in tqdm(range(len(data))):
        input = data.iloc[i]['input']

        indexed_tokens = input['indexed_tokens'].to('cuda')
        segment_ids = input['segment_ids'].to('cuda')

        with torch.no_grad():
            prediction = model(indexed_tokens, segment_ids) 
            activations.append(prediction.cpu())

    data['output'] = activations
    return data

# Split a list into n approximately equal chunks
# from https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# Tokenize and split a document into BERT inputs
def split_doc(query, doc, tokenizer, at_period = False) -> list:
    inputs = []
    max_input_size = tokenizer.max_len

    all_doc_tokens = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc)))
    doc_length = len(all_doc_tokens)

    query_tokens = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS] ' + query + ' [SEP]')))

    # get indices of [SEP]-tokens, which indicate start and end of encoded document
    sep_token = 102
    part1 = query_tokens

    # Number of token slots available for document
    doc_tokens_available = max_input_size - len(query_tokens) - 2
    
    # Starting index of current document slice
    slice_start = 0
    tokens_left = True
    while tokens_left:
        # slice_start = slice_end +1 = last_period +1 (after first iteration)
        slice_end = min(doc_length, slice_start + doc_tokens_available)
        if slice_start >= slice_end:
            break

        # If remaining text too long, split at period
        if at_period and not slice_end == doc_length:
            period_token = 1012
            period_indices = (all_doc_tokens[slice_start : slice_end] == period_token).nonzero()
            
            if not period_indices.size()[0] == 0:
                last_period = slice_start + period_indices[-1][0].item()
                slice_end = last_period

        split_size = slice_end - slice_start
        
        doc_slice = all_doc_tokens[ slice_start : slice_end + 1] # including end index
        #print(f"  Doc slice: {doc_slice[:4], doc_slice[-4:]}")
        input = {}
        part2 = doc_slice
        part3 = torch.LongTensor([sep_token])

        # Change input with new slice
        input['indexed_tokens'] = torch.unsqueeze(torch.cat((part1, part2, part3)), 0)

        # Change input['segment_ids']
        input['segment_ids'] = torch.unsqueeze(torch.cat( (torch.zeros((len(query_tokens) - 1), dtype=torch.int64),
                                              torch.ones((len(doc_slice)+2), dtype=torch.int64) )), 0)

        inputs.append(input)
        slice_start = slice_end + 1  # should start at token after last period

    if len(inputs) == 0:
        inputs = [{
            'indexed_tokens': torch.unsqueeze(torch.cat((part1, torch.LongTensor([sep_token]))), 0),
            'segment_ids': torch.unsqueeze(torch.cat( (torch.zeros((len(query_tokens) - 1), dtype=torch.int64),
                                              torch.ones((2), dtype=torch.int64) )), 0)

        }]

    return inputs