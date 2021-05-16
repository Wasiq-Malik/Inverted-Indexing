import os
import json
import time
from heapq import merge
from bs4 import BeautifulSoup
from nltk.stem import snowball
from bs4.element import Comment
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize

docID = 0


def preprocess_docs(dir):
    global docID
    corpus = []
    doc_info = {}
    start_id = docID

    files_count = len(os.listdir(dir))
    print(
        f"\n[Preprocessor] Processing directory {dir} with {files_count} files ... ")

    for i, filename in enumerate(os.listdir(dir)):
        file_path = os.path.join(dir, filename)

        # open in readonly mode
        with open(file_path, 'r', encoding='utf-8') as html_doc:

            start_time = time.time()

            try:
                text = extract_text(html_doc.read())

                all_tokens, tokens = tokenize_text(text)

                tokens = stop_wording(tokens)

                tokens = stemming(tokens)

                word_pos = word_positions(all_tokens, tokens)

            except Exception as e:
                print(
                    f"[Preprocessor/{filename.split('.')[0]}] processing failed with error {e}.")
                continue

            end_time = time.time()

            docID += 1

            doc_info[docID] = {"length": len(
                all_tokens), "magnitude": calculate_magnitude(word_pos), "path": file_path}
            corpus.append([docID, tokens, word_pos])

            print(
                f"[Preprocessor/{filename.split('.')[0]}] {len(all_tokens)} tokens, processed in {(end_time - start_time):.3f} seconds.")

    print(f"[Preprocessor] Processed {docID - start_id} files in {dir}. ")

    return corpus, doc_info


def extract_text(doc):

    # tells if an html is visible on the web page or not
    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    # ignoring https headers from html file
    if doc.find("<!DOCTYPE") > 0:
        doc = doc[doc.find("<!DOCTYPE"):]

    # parse html and only keep visible web page text
    soup = BeautifulSoup(doc, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)

    return u" ".join(t.strip() for t in visible_texts)


def tokenize_text(text):

    # generate word tokens
    tokens = []
    for sentence in sent_tokenize(text):
        word_tokens = word_tokenize(sentence)
        tokens += word_tokens

    # complete doc tokenized
    all_tokens = [x.lower() for x in tokens]

    # only keep unique words
    unique_words = set([x.lower() for x in tokens if x.encode().isalpha()])

    return all_tokens, unique_words


def stop_wording(tokens):

    stop_words = stopwords.words('english')

    # set difference to remove stop words
    return tokens - set(stop_words)


def stemming(tokens):

    stemmer = snowball.SnowballStemmer('english')

    # keep unique stems
    stemmed_words = set([stemmer.stem(x) for x in tokens])

    return stemmed_words


def word_positions(all_tokens, stemmed_words):

    pos_dict = dict()

    stemmer = snowball.SnowballStemmer('english')

    # calculating positions for each unique word
    for i, token in enumerate(all_tokens):
        stemmed_token = stemmer.stem(token)

        # check if token is a word
        if stemmed_token in stemmed_words:
            # add current position in posting list
            if stemmed_token in pos_dict:
                pos_dict[stemmed_token].append(i)
            else:
                pos_dict[stemmed_token] = [i]

    return pos_dict


def calculate_magnitude(word_pos):

    # sum of squares of word frequencies
    running_squares = 0
    for key in word_pos.keys():
        running_squares += (len(word_pos[key])) ** 2

    return round(running_squares ** (1/2), 2)  # sqrt


def create_inverted_index(corpus):

    inverted_index = defaultdict(list)

    # generate inverted index with doc IDs and word positions
    for [docID, words, positions] in corpus:
        for token in list(words):
            word_pos = positions[token]
            inverted_index[token].append(
                {'id': docID, 'freq': len(word_pos), 'pos': word_pos})

    # sort the index
    inverted_index = dict(sorted(inverted_index.items()))

    return inverted_index


def store_inverted_index(inverted_index, directory_name):

    posting_file = open(
        f"index_{directory_name}_postings.txt", encoding='utf-8',  mode='w')
    byte_index = 0

    # write index and posting list in text files
    with open(f"index_{directory_name}_terms.txt", encoding='utf-8', mode='w') as terms_file:
        for token in inverted_index.keys():
            terms_file.write(f"{token}, {byte_index}\n")

            byte_index += write_posting_list(posting_file,
                                             inverted_index[token])

    print(
        f"[Inverter] Index for directory '{directory_name}' successfully created.")


def read_index_words(path):

    vocab = {}
    index_file = open(path, encoding='utf-8', mode='r')

    line = index_file.readline()
    while line != '':

        # insert word:byte_location pair in look-up table
        token = line.split(', ')
        vocab[token[0]] = int(token[1])
        line = index_file.readline()

    return vocab


def write_posting_list(file_pointer, posting_list):
    byte_index = 0

    # get document frequency for the token
    df = len(posting_list)

    byte_index += file_pointer.write(f"{df},")  # write doc freq in bytes

    # write posting list for current token in posting file
    for occurance in posting_list:
        doc_id = occurance["id"]
        frequency = occurance["freq"]
        positions = occurance["pos"]

        # write document id in bytes
        byte_index += file_pointer.write(f"{doc_id},")

        # write word frequency in bytes
        byte_index += file_pointer.write(f"{frequency},")

        # delta encode the positions if occurance frequency more than once
        positions = [positions[0]] + [positions[i]-positions[i-1]
                                      for i in range(1, frequency)] if frequency > 1 else [positions[0]]

        # write position values in bytes
        for pos in positions:
            byte_index += file_pointer.write(f"{pos},")

    byte_index += file_pointer.write("\n")

    return byte_index + 1


def read_posting_list(file_pointer):

    posting_list = list()
    line = file_pointer.readline()
    tokens = [int(tok) for tok in line.split(",") if tok.isnumeric()]

    idx = 1
    for _ in range(tokens[0]):
        posting = {"id": tokens[idx]}   # read off doc id
        idx += 1
        posting["freq"] = tokens[idx]   # read off word freq
        idx += 1
        posting["pos"] = [tokens[idx]]  # read off first position
        idx += 1

        for i in range(posting["freq"] - 1):
            # revert delta encoding
            posting["pos"].append(posting["pos"][i] + tokens[idx])
            idx += 1

        posting_list.append(posting)

    return posting_list


def merge_posting_lists(list1, list2):
    return list(merge(list1, list2, key=lambda x: x["id"]))


def merge_sorted_indexes(directory_names):

    print(f"[Merger] Merging indexes from each directory ...")

    byte_index = 0
    unique_words = []

    index_pointers = [open(
        f"index_{name}_terms.txt", encoding='utf-8', mode='r') for name in directory_names]
    posting_list_pointers = [open(
        f"index_{name}_postings.txt", encoding='utf-8', mode='r') for name in directory_names]

    index_file = open("inverted_index_terms.txt", encoding='utf-8', mode='w')
    posting_file = open("inverted_index_postings.txt",
                        encoding='utf-8', mode='w')

    # fetch first words from each file
    index_lines = {file_id: ptr.readline()
                   for file_id, ptr in enumerate(index_pointers)}

    # loop till unmerged files remain
    while len(index_lines):

        # extract (word, byte pos) pair from current line in index file
        index_words = [(file_id, index_lines[file_id].split(", ")[0], int(
            index_lines[file_id].split(", ")[1])) for file_id in index_lines.keys()]

        # find min word to merge
        min = index_words[0][1]
        for _, word, _ in index_words:
            if word < min:
                min = word

        # save mins if more of same word found
        mins = [(file_id, word, byte)
                for (file_id, word, byte) in index_words if word == min]
        unique_words.append(min)

        # merge next (word, posting list) pair and increment respective index files pointers
        merged_posting = []
        for (file_id, word, byte) in mins:

            # retrieve posting list from file to merge
            posting_list_pointers[file_id].seek(byte, 0)
            curr_posting = read_posting_list(posting_list_pointers[file_id])

            # merge the lists
            merged_posting = merge_posting_lists(merged_posting, curr_posting)

            # fetch the next
            index_lines[file_id] = index_pointers[file_id].readline()

        # write current min word in merged index
        index_file.write(f"{min}, {byte_index}\n")

        # write min word's posting list in merged posting
        byte_index += write_posting_list(posting_file, merged_posting)

        # check if any index file as been completely merged, if so remove it
        index_lines = {key: index_lines[key]
                       for key in index_lines.keys() if index_lines[key] != ''}

    return unique_words


def store_docs_info(docs_info):
    docs_info_file = open("docs_meta_data.txt", encoding='utf-8', mode='w')
    docs_info_file.write(json.dumps(docs_info))
    docs_info_file.close()


def load_docs_info(path):
    docs_info_file = open(path, encoding='utf-8', mode='r')
    return json.loads(docs_info_file.read())


def boolean_retrieval(query, vocab, docs_info):

    print(f"[Boolean-Retriever] Querying '{query}' against index ... ")
    start_time = time.time()

    # tokenize query
    tokens = []
    for sentence in sent_tokenize(query):
        word_tokens = word_tokenize(sentence)
        tokens += word_tokens

    # clean query
    tokens = [tok.lower() for tok in tokens if tok.encode().isalpha()]

    # stop word removal
    stop_words = stopwords.words('english')
    tokens = list(set(tokens) - set(stop_words))

    # stemming query tokens
    stemmer = snowball.SnowballStemmer('english')
    tokens = [stemmer.stem(tok) for tok in tokens]

    query_matches = set()
    with open("inverted_index_postings.txt", encoding='utf-8', mode='r') as posting_file:
        for word in tokens:
            if word in vocab:
                posting_loc = vocab[word]
                posting_file.seek(posting_loc)
                posting_list = read_posting_list(posting_file)

                for posting in posting_list:
                    doc_id = str(posting["id"])
                    query_matches.add(docs_info[doc_id]["path"])

    if len(query_matches):
        list(query_matches).sort()
        for match in query_matches:
            print(f"[Boolean-Retriever] match found at '{match}'")

        end_time = time.time()
        print(f"[Boolean-Retriever] Found {len(query_matches)} matches in {(end_time - start_time):.3f} seconds.")
    else:
        print("[Boolean-Retriever] No Match Found.")


def multi_query_test(queries):

    docs_info = load_docs_info('docs_meta_data.txt')

    vocab = read_index_words('inverted_index_terms.txt')

    for query in queries:
        boolean_retrieval(query, vocab, docs_info)

def build_complete_index(base_path):
    directories = os.listdir(base_path)

    docs_info = {}  # accumulate docs meta data from each directory
    for dir in directories:
        # building inverted index on each directory
        corpus, doc_info = preprocess_docs(os.path.join(base_path, dir))
        inverted_index = create_inverted_index(corpus)
        store_inverted_index(inverted_index, dir)
        docs_info.update(doc_info)

    store_docs_info(docs_info)

    print(
        f"\n[Merger] Found {len(merge_sorted_indexes(directories))} unique words in Merged Index.\n")

if __name__ == "__main__":

    # index building
    base_path = "D:\\Semester Work\\IR Indexing\\corpus\\corpus1\\"
    #build_complete_index(base_path)

    # boolean retrieval
    query = ["Information Retrieval CS"]
    multi_query_test(query)
