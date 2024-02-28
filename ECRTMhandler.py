import torch
from torch.utils.data import DataLoader
import numpy as np
import gensim
from topmost.preprocessing import Preprocessing as tmPreprocessings
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

class DataHandler:
    def __init__(self, bow, batch_size=200, device='cpu', as_tensor=False, vocab=[]):
        if device != "cpu":
            torch.cuda.empty_cache()
        print("cache clear")
        # train_bow: NxV
        # test_bow: Nxv
        # word_emeddings: VxD
        # vocab: V, ordered by word id.
        self.train_bow, self.test_bow = train_test_split(bow, test_size=0.2)
        # self.train_bow = bow
        # self.test_bow = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
        # self.pretrained_WE = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')
        # for text in parsed_texts:
        #     vocab = vocab.union(text.split())
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        print("===>train_size: ", self.train_bow.shape[0])
        print("===>test_size: ", self.test_bow.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        if as_tensor:
            self.train_bow = torch.from_numpy(self.train_bow).to(device, dtype=torch.float32)
            self.test_bow = torch.from_numpy(self.test_bow).to(device, dtype=torch.float32)
            self.train_dataloader = DataLoader(self.train_bow, batch_size=batch_size, shuffle=True)
            self.test_dataloader = DataLoader(self.test_bow, batch_size=batch_size, shuffle=False)
        if device != "cpu":
            torch.cuda.empty_cache()

class Preprocessings(tmPreprocessings):
    def make_word_embeddings(self, vocab):
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
        word_embeddings = np.zeros((len(vocab), glove_vectors.vectors.shape[1]))

        num_found = 0
        for i, word in enumerate(tqdm(vocab, desc="===>making word embeddings")):
            try:
                key_word_list = glove_vectors.index_to_key
            except:
                key_word_list = glove_vectors.index2word

            if word in key_word_list:
                word_embeddings[i] = glove_vectors[word]
                num_found += 1

        print(f'===> number of found embeddings: {num_found}/{len(vocab)}')

        return sparse.csr_matrix(word_embeddings)
    
    def parse(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
            
        n_items = len(texts)
        print(f"Found training documents {n_items}")

        parsed_texts = list()
        # word_counts = Counter()
        doc_counts_counter = Counter()

        for _, text in enumerate(tqdm(texts, desc="===>parse texts")):

            # tokens = tokenize(text, strip_html=self.strip_html, lower=(not self.no_lower), keep_numbers=self.keep_num, keep_alphanum=self.keep_alphanum, min_length=self.min_length, stopwords=self.stopword_set)
            tokens = self.tokenizer.tokenize(text)
            # word_counts.update(tokens)
            doc_counts_counter.update(set(tokens))
            # train_texts and test_texts have been parsed.
            parsed_texts.append(' '.join(tokens))

        words, doc_counts = zip(*doc_counts_counter.most_common())
        doc_freqs = np.array(doc_counts) / float(n_items)
        vocab = [word for i, word in enumerate(words) if doc_counts[i] >= self.min_doc_count and doc_freqs[i] <= self.max_doc_freq]

        # filter vocabulary
        if (self.vocab_size is not None) and (len(vocab) > self.vocab_size):
            vocab = vocab[:self.vocab_size]

        vocab.sort()

        print(f"Real vocab size: {len(vocab)}")
        print("===>convert to matrix...")
        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
        bow_matrix = vectorizer.fit_transform(parsed_texts)
        bow_matrix = bow_matrix.toarray()
        word_embeddings = self.make_word_embeddings(vocab)
        word_embeddings = word_embeddings.toarray()

        return parsed_texts, bow_matrix, vocab, word_embeddings
