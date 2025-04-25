#!/usr/bin/env python3

"""
Disclaimer: There is a BIG BIG BIG bug in the code that causes it to not work as expected.
Please refere to the tokenize_query function for the detailed expalination.
"""

"""
Vector Space Model with BM25 Implementation for Information Retrieval
Used for NTCIR Chinese news article retrieval with Rocchio relevance feedback.

This script implements a BM25-based information retrieval system that:
1. Loads model files (vocabulary, document list, inverted index)
2. Processes queries from XML files
3. Retrieves relevant documents using BM25 ranking
4. Supports Rocchio relevance feedback (pseudo version)
5. Evaluates retrieval performance using Mean Average Precision (MAP)
"""

import os
import sys
import math
import time
import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter


class VSM_BM25:
    """
    Vector Space Model with BM25 ranking implementation.
    
    Attributes:
        model_dir (str): Directory containing model files
        ntcir_dir (str): Directory containing NTCIR documents
        vocab (list): List of vocabulary terms
        file_list (list): List of document file paths
        doc_term_freq (dict): Document-term frequency matrix {doc_id: {term_id: freq}}
        term_doc_freq (dict): Term-document frequency matrix {term_id: {doc_id: freq}}
        idf (dict): Inverse document frequency values {term_id: idf_value}
        doc_lengths (dict): Document lengths {doc_id: length}
        avg_doc_length (float): Average document length
        N (int): Total number of documents
        k1, b, k3 (float): BM25 parameters
    """
    
    def __init__(self, model_dir, ntcir_dir):
        """
        Initialize the VSM_BM25 model.
        
        Args:
            model_dir (str): Directory containing model files
            ntcir_dir (str): Directory containing NTCIR documents
        """
        self.model_dir = model_dir
        self.ntcir_dir = ntcir_dir
        self.vocab = []
        self.file_list = []
        self.doc_term_freq = {}  # {doc_id: {term_id: freq}}
        self.term_doc_freq = {}  # {term_id: {doc_id: freq}}
        self.idf = {}            # {term_id: idf_value}
        self.doc_lengths = {}    # {doc_id: doc_length}
        self.avg_doc_length = 0  # Average document length
        self.N = 0               # Total number of documents
        
        # BM25 parameters (please refer to PAGE 48 of vsmodel_2025.pdf for the value references) )
        self.k1 = 1.2   # Term frequency saturation parameter : between 1.0 and 2.0
        self.b = 0.75   # Length normalization parameter : usually 0.75
        self.k3 = 8     # Query term weight parameter : between 0 and 1000
        
        # Load model files
        self._load_model()
        
    def _load_model(self):
        """
        Load model files (vocab.all, file-list, inverted-file) under "wm-2025-vector-space-mode/model"
        
        This method loads:
        1. Vocabulary from vocab.all: 29907 terms in the vocabulary.
        2. Document list from file-list: 46972 documents.
        3. Term-document frequencies from inverted-file : 1,193,467 terms: consisting of 22,220 unigrams and 1,171,247 bigrams.
        
        It also calculates document lengths and BM25 IDF values.
        """
        start_time = time.time()
        print("Loading model files...")
        
        ### 1. HW Slide / NTCIR Document Set / vocab.all ###
        # As per vsmodel_2025.pdf / PAGE 48, we should use techniques mentioned in "A. Singhal, "Modern Information Retrieval: A Brief Overview,IEEE Data Engineering Bulletin, vol. 24(4), pp. 35-43, 2001", like removing stop words, stemming.
        #  But in this homework, we just use the vocab.all as it is.
        #  This may explain why my model performance stuck around 0.76 MAP.
        #  Need to do the preprocessing in the future.
        vocab_path = os.path.join(self.model_dir, "vocab.all")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = f.read().splitlines()
        
        # The first line is character encoding format (utf8), ignore it
        self.vocab = self.vocab[1:]
        
        ### 2. HW Slide / NTCIR Document Set / file-list ###
        file_list_path = os.path.join(self.model_dir, "file-list")
        with open(file_list_path, 'r', encoding='utf-8') as f:
            self.file_list = f.read().splitlines()
        
        self.N = len(self.file_list)
        
        ### 3. HW Slide / NTCIR Document Set / inverted-file ###
        inverted_file_path = os.path.join(self.model_dir, "inverted-file")
        self._load_inverted_file(inverted_file_path)
        
        ### vsmodel_2025.pdf / PAGE 24 / Okapi/BM25 Doc Length Normalization ###
        # Calculate document lengths and average document length
        self._calculate_doc_lengths()
        
        ### vsmodel_2025.pdf / PAGE 48 / What works the best? ###
        # First term in the summation.
        self._calculate_bm25_idf()
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total documents: {self.N}")
        print(f"Average document length: {self.avg_doc_length:.2f}")
    
    def _load_inverted_file(self, inverted_file_path):
        """
        Homework Description
            1. Total terms: 1193467 (Unigrams: 22220, Bigrams: 1171247)
            2. vocab_id and file_id referred from vocab.all and file-list. 
            3. vocab_id_1 vocab_id_2 denotes an unigram when vocab_id_2==-1
            4. vocab_id_1 vocab_id_2 denote a bigram when vocab_id_2!=-1. 
            5. If there are N files containing vocab_id_1 vocab_id_2, there will be the number N next to vocab_id_2, followed by N lines that display the counts of this term in each file
            vocab_id_1  vocab_id_2  N
            file_id  count … (N lines)
        Function
            Load and process the inverted file containing term-document frequencies.
        Args:
            inverted_file_path (str): Path to the inverted file
        """
        with open(inverted_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        line_index = 0
        while line_index < len(lines):
            parts = lines[line_index].strip().split()
            vocab_id_1 = int(parts[0])
            vocab_id_2 = int(parts[1])
            N = int(parts[2])  # Number of files containing this term
            
            # Create a unique term ID
            if vocab_id_2 == -1:  # Unigram
                term_id = f"u{vocab_id_1}"
            else:  # Bigram
                term_id = f"b{vocab_id_1}_{vocab_id_2}"
            
            # Initialize term-doc frequency dictionary for this term
            self.term_doc_freq[term_id] = {}
            
            # Read the next N lines for file_id and count
            for i in range(N):
                line_index += 1
                file_parts = lines[line_index].strip().split()
                file_id = int(file_parts[0])
                count = int(file_parts[1])
                
                # Add to term-doc frequency
                self.term_doc_freq[term_id][file_id] = count
                
                # Add to doc-term frequency
                if file_id not in self.doc_term_freq: ## { key = doc_id: value = {term_id: freq} }
                    self.doc_term_freq[file_id] = {}
                self.doc_term_freq[file_id][term_id] = count
            
            line_index += 1
    
    def _calculate_doc_lengths(self):
        """
        Calculate document lengths and average document length.
        
        Document length is defined as the sum of term occurrences/frequencies in the document.
        Average document length is used in BM25 for document length normalization.
        """
        total_length = 0
        
        for doc_id, terms in self.doc_term_freq.items():
            # Sum of term frequencies in this document
            doc_length = sum(terms.values())
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length
        
        # Calculate average document length
        if self.N > 0:
            self.avg_doc_length = total_length / self.N
    
    ### vsmodel_2025.pdf / PAGE 48 / What works the best?
    def _calculate_bm25_idf(self):
        """
        Calculate IDF values for BM25 formula.
        
        BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1.0)
        where N is the total number of documents and df is the document frequency.
        """
        for term_id, docs in self.term_doc_freq.items():
            df = len(docs)  # Document frequency
            # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5))
            self.idf[term_id] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
    
    def parse_query(self, query_file):
        """
        Parse query XML file and extract query information.
        
        Args:
            query_file (str): Path to the query XML file
            
        Returns:
            list: A list of tuples (query_id, query_text) for each query topic
        Query File Format:
            1. The NTCIR topic format conforms to XML 1.0, in which the document is rooted at an <xml> tag.
            2. The file contains multiple topics, each of them is enclosed in a <topic> tag. In each topic, different types of information are specified by the following tags:
            2-1. <number>: The topic number.
            2-2. <title>: The topic title.
            2-3. <question>: A short description about the query topic.
            2-4. <narrative>: Even more verbose descriptions about the topic.
            2-5. <concepts>: A set of keywords that can be used in retrieval about the topic.
            3. You have to retrieve several relevant documents for each topic.
            4. All the content of title, question, narrative, and concepts can be used as the query of the topic, it's your own choice to decide which part(s) you want to use.
        """
        tree = ET.parse(query_file)
        root = tree.getroot()
        
        queries = []
        for topic in root.findall(".//topic"):
            query_id = topic.find("number").text[-3:]  # Last 3 digits
            
            # Extract all text from relevant fields
            title = topic.find("title").text if topic.find("title") is not None else ""
            question = topic.find("question").text if topic.find("question") is not None else ""
            concepts = topic.find("concepts").text if topic.find("concepts") is not None else ""
            
            # In an ideal scenario, we should learn the weighting for title, question, and concepts. (did not do that this time.)
            # i.e. based on the training performance MAP to fine-tune the hyper parameters.
            # For example, title * 2 + question * 1 + concepts * 0.5
            # But in this homework, I just concatenate them as the query text.
            query_text = f"{title} {question} {concepts}"
            
            queries.append((query_id, query_text))
        
        return queries
    
    def tokenize_query(self, query_text):
        """
        Tokenize query text into unigrams and bigrams that exist in vocabulary.
        
        Args:
            query_text (str): Query text to tokenize
            
        Returns:
            Counter: Term counts for query terms
        Comments:
            1. For Chinese, we split the query into characters (unigrams).
            2. For English, we split the query into words.
            3. We only keep unigrams and bigrams that exist in the vocabulary.
            4. For bigrams, we check if both characters exist in the vocabulary.
            5. We create a term ID for unigrams and bigrams based on their vocabulary IDs.
            6. Note that we do not consider N-grams (N>2) in this homework as I asked TA whether using the inverted file is enough, and got an affirmative answer.
        """
        # Later found out that this is a BIG BUG in my code. The following is a counterexample in which the code will render wrong results.
        # query_string="Camus勸告我們不要進行哲學自殺及其相應的信仰飛躍，這樣我們才能保持理性直到生命的最後一刻。"
        # query_list=list(query_string)
        # query_list
        # ['C',
            # 'a',
            # 'm',
            # 'u',
            # 's',
            # '勸',
            # '告',
            # '我',
            # '們',
            # '不',
            # '要',
            # '進',
            # '行',
            # '哲',
            # '學',
            # '自',
            # '殺',
            # '及',
            # '其',
            # '相',
            # '應',
            # '的',
            # '信',
            # '仰',
            # '飛',
            # '躍',
            # '，',
            # '這',
            # '樣',
            # '我',
            # '們',
            # '才',
            # '能',
            # '保',
            # '持',
            # '理',
            # '性',
            # '直',
            # '到',
            # '生',
            # '命',
            # '的',
            # '最',
            # '後',
            # '一',
            # '刻',
            # '。']
        # Initially, I thought the query is a string, an since we only need to consider unigram and bigram so I can simply split the string into characters.
        # But I did not take into account that the query may contain both Chinese and English words.
        # So I need to split the query into characters (for Chinese) and words (for English), so using NLTK or jieba is a MUCH MUCH better choice. sigh....
        # Split query into characters (for Chinese) and words (for English)
        query_chars = list(query_text)
        
        # Create unigrams that exist in vocabulary
        unigrams = []
        for char in query_chars:
            if char in self.vocab:
                vocab_id = self.vocab.index(char) + 1  # +1 because vocab_id starts from 1
                term_id = f"u{vocab_id}"
                if term_id in self.term_doc_freq:
                    unigrams.append(term_id)
        
        # Create bigrams that exist in vocabulary
        bigrams = []
        for i in range(len(query_chars) - 1):
            char1 = query_chars[i]
            char2 = query_chars[i + 1]
            
            if char1 in self.vocab and char2 in self.vocab:
                vocab_id_1 = self.vocab.index(char1) + 1
                vocab_id_2 = self.vocab.index(char2) + 1
                term_id = f"b{vocab_id_1}_{vocab_id_2}"
                if term_id in self.term_doc_freq:
                    bigrams.append(term_id)
        
        # Combine unigrams and bigrams
        terms = unigrams + bigrams
        term_counts = Counter(terms)
        
        return term_counts
    
    def compute_query_weights(self, term_counts):
        """
        Compute query term weights using BM25's query term frequency component.
        
        Args:
            term_counts (Counter): Term frequency counts in the query
            
        Returns:
            dict: Query term weights {term_id: weight}
        
        Comments:
            the formula can be inferred from the vsmodel_2025.pdf / PAGE 48 / What works the best? section.
        """
        query_weights = {}
        
        for term_id, count in term_counts.items():
            if term_id in self.idf:  # Only consider terms in our vocabulary (bigrams and unigrams)
                # BM25 query term weight: ((k3 + 1) * qtf) / (k3 + qtf)
                query_weight = ((self.k3 + 1) * count) / (self.k3 + count)
                # Multiply by IDF
                query_weights[term_id] = query_weight * self.idf[term_id]
        
        return query_weights
    
    def compute_bm25_score(self, query_weights, doc_id):
        """
        Compute BM25 score between query and document.
        
        BM25 score = sum_over_terms(IDF * ((k1+1)*tf) / (k1*((1-b) + b*dl/avgdl) + tf) * ((k3+1)*qtf) / (k3+qtf))
        
        Args:
            query_weights (dict): Query term weights
            doc_id (int): Document ID
            
        Returns:
            float: BM25 score
        
        Comments:
            the formula can be inferred from the vsmodel_2025.pdf / PAGE 48 / What works the best? section.
        """
        if doc_id not in self.doc_term_freq:
            return 0
        
        doc_terms = self.doc_term_freq[doc_id]
        doc_length = self.doc_lengths[doc_id]
        
        # BM25 score accumulator
        score = 0
        
        for term_id, query_weight in query_weights.items():
            if term_id in doc_terms:
                tf = doc_terms[term_id]
                
                # BM25 document term weight component (this is just the middle term on the slide 48 of vsmodel_2025.pdf):
                # ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (dl/avgdl)) + tf)
                numerator = (self.k1 + 1) * tf
                denominator = self.k1 * ((1 - self.b) + self.b * (doc_length / self.avg_doc_length)) + tf
                doc_weight = numerator / denominator
                
                # Add to total score
                score += query_weight * doc_weight
        
        return score
    
    def retrieve_documents(self, query_weights, score_threshold=0.1, max_docs=100):
        """
        Retrieve documents with BM25 score above threshold, up to max_docs.
        
        Args:
            query_weights (dict): Query term weights
            score_threshold (float): Minimum score threshold
            max_docs (int): Maximum number of documents to retrieve
            
        Returns:
            list: List of tuples (doc_id, score) for retrieved documents
        """
        scores = []
        
        for doc_id in range(self.N):
            score = self.compute_bm25_score(query_weights, doc_id)
            # we just run several experiments on the training queries to find out the optimal threshold value.
            if score > score_threshold:
                scores.append((doc_id, score))
        
        # Sort by score in descending order as required by the "Ranking List Format" in the homework slide.
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top documents (up to max_docs) (homework description : You can retrieve at most 100 documents for each topic. )
        return scores[:max_docs]
    
    def get_document_id(self, file_id):
        """
        Get document ID from file path.
        
        Args:
            file_id (int): File ID
            
        Returns:
            str: Document ID (lowercase)
        Comments:
            1. The document ID is the last part of the file path (e.g., CDN_LOC_0001457).
            2. The file path is in the format:
                CIRB010/cdn/loc/CDN_LOC_0001457
                CIRB010/cdn/loc/CDN_LOC_0000294
                CIRB010/cdn/loc/CDN_LOC_0000120
            3. The document ID is converted to lowercase.
        """
        file_path = self.file_list[file_id]
        # Extract the last part of the path (e.g., CDN_LOC_0001457)
        doc_id = os.path.basename(file_path).lower()
        return doc_id
    
    def rocchio_feedback(self, query_weights, relevant_docs, irrelevant_docs=None, alpha=1.0, beta=0.75, gamma=0.15):
        """
        Implement Rocchio algorithm for query expansion with BM25 weights.
        
        Modified query = α * original_query + β/|Dr| * sum(relevant_docs) - γ/|Dn| * sum(non_relevant_docs)
        
        Args:
            query_weights (dict): Original query weights
            relevant_docs (list): List of relevant document IDs
            irrelevant_docs (list): List of irrelevant document IDs
            alpha (float): Weight for original query
            beta (float): Weight for relevant documents
            gamma (float): Weight for irrelevant documents
            
        Returns:
            dict: Modified query weights
        
        Comments:
            Page 42 of vsmodel_2025.pdf / Rocchio Algorithm
            1. α: weight for original query
            2. β: weight for relevant documents
            3. γ: weight for irrelevant documents
            4. |Dr|: number of relevant documents
            5. |Dn|: number of irrelevant documents
            6. The Rocchio algorithm is used to refine the query based on feedback from relevant and irrelevant documents.
        """
        if not relevant_docs:
            return query_weights
        
        # Initialize the modified query with the original query
        modified_query = {term: weight * alpha for term, weight in query_weights.items()}
        
        # Add contribution from relevant documents
        # Please refer to my impelementation of the retrieve_documents function.
        # The top k is replaced by the score threshold.
        # So for each query, the number of relevant_docs may vary. (This can also be discussed further).
        num_relevant = len(relevant_docs)
        if num_relevant > 0:
            for doc_id in relevant_docs:
                if doc_id in self.doc_term_freq:
                    doc_terms = self.doc_term_freq[doc_id]
                    doc_length = self.doc_lengths[doc_id]
                    
                    for term_id, freq in doc_terms.items():
                        # Calculate BM25 term weight for this document
                        numerator = (self.k1 + 1) * freq
                        denominator = self.k1 * ((1 - self.b) + self.b * (doc_length / self.avg_doc_length)) + freq
                        doc_weight = numerator / denominator * self.idf[term_id]
                        
                        ### Term Re-weighting (page 42 of vsmodel_2025.pdf) ###
                        # Add to modified query
                        # α * original_query + β/|Dr| * sum(relevant_docs)  
                        # Note: We use beta / num_relevant to normalize the contribution from relevant documents
                        # to the modified query.
                        # This is the same as the formula on page 42 of vsmodel_2025.pdf
                        # where β/|Dr| is the weight for relevant documents.
                        if term_id in modified_query:
                            modified_query[term_id] += (beta / num_relevant) * doc_weight
                        else:
                            modified_query[term_id] = (beta / num_relevant) * doc_weight
        
        # Subtract contribution from irrelevant documents
        # As per page 44 of vsmodel_2025.pdf, negative (non-relevant) examples are not very important.
        # So we in fact ignore (set it as None) the irrelevant documents in the Rocchio algorithm.
        # But I still implement it here for the sake of completeness.
        if irrelevant_docs:
            num_irrelevant = len(irrelevant_docs)
            if num_irrelevant > 0:
                for doc_id in irrelevant_docs:  
                    if doc_id in self.doc_term_freq:
                        doc_terms = self.doc_term_freq[doc_id]
                        doc_length = self.doc_lengths[doc_id]
                        
                        for term_id, freq in doc_terms.items():
                            # Calculate BM25 term weight for this document
                            numerator = (self.k1 + 1) * freq
                            denominator = self.k1 * ((1 - self.b) + self.b * (doc_length / self.avg_doc_length)) + freq
                            doc_weight = numerator / denominator * self.idf[term_id]
                            
                            if term_id in modified_query:
                                modified_query[term_id] -= (gamma / num_irrelevant) * doc_weight
        
        # Remove terms with negative weights
        modified_query = {t: w for t, w in modified_query.items() if w > 0}
        
        return modified_query
    
    def pseudo_relevance_feedback(self, query_weights, top_docs=10, alpha=1.0, beta=0.75, num_iterations=1):
        """
        Implement pseudo relevance feedback with multiple iterations.
        
        In pseudo relevance feedback, the top-ranked documents from initial retrieval
        are assumed to be relevant and used for query expansion.
        
        Args:
            query_weights (dict): Original query weights
            top_docs (int): Number of top documents to consider as relevant
            alpha (float): Weight for original query
            beta (float): Weight for relevant documents
            num_iterations (int): Number of feedback iterations
            
        Returns:
            dict: Modified query weights
        """
        current_query = query_weights
        
        for iteration in range(num_iterations):
            # Retrieve initial results using current query
            initial_results = self.retrieve_documents(current_query, max_docs=top_docs)
            
            # Consider top k documents as relevant (This is why it is called "PSUEDO" relevance feedback)
            # If fact, in my implementation (retrieve_documents), the top k is replaced by the score threshold.
            # So for each query, the number of relevant_docs may vary. (This can also be discussed further).
            relevant_docs = [doc_id for doc_id, _ in initial_results]
            
            # Apply Rocchio algorithm
            current_query = self.rocchio_feedback(current_query, relevant_docs, None, alpha, beta, 0)
        
        return current_query
    
    def process_query(self, query_id, query_text, use_feedback=False, feedback_iterations=1, 
                     alpha=1.0, beta=0.75, gamma=0.15, score_threshold=0.1):
        """
        Process a query and return ranked documents.
        
        Args:
            query_id (str): Query ID
            query_text (str): Query text
            use_feedback (bool): Whether to use relevance feedback
            feedback_iterations (int): Number of feedback iterations
            alpha (float): Weight for original query
            beta (float): Weight for relevant documents
            gamma (float): Weight for irrelevant documents
            score_threshold (float): Minimum score threshold
            
        Returns:
            tuple: (query_id, results) where results is a list of (doc_id, score) tuples
        """
        # Tokenize query and compute BM25 weights
        term_counts = self.tokenize_query(query_text)
        query_weights = self.compute_query_weights(term_counts)
        
        # Apply pseudo relevance feedback if enabled
        if use_feedback:
            query_weights = self.pseudo_relevance_feedback(
                query_weights, top_docs=10, alpha=alpha, beta=beta, 
                num_iterations=feedback_iterations
            )
        
        # Retrieve documents above score threshold
        ranked_docs = self.retrieve_documents(query_weights, score_threshold, max_docs=100)
        
        # Convert file IDs to document IDs
        results = [(self.get_document_id(doc_id), score) for doc_id, score in ranked_docs]
        
        return query_id, results
    
    def process_queries(self, query_file, output_file, use_feedback=False, feedback_iterations=1, 
                       alpha=1.0, beta=0.75, gamma=0.15, score_threshold=0.1):
        """
        Process all queries and write results to output file for uploading to Kaggle.
        
        Args:
            query_file (str): Path to query file
            output_file (str): Path to output file
            use_feedback (bool): Whether to use relevance feedback
            feedback_iterations (int): Number of feedback iterations
            alpha (float): Weight for original query
            beta (float): Weight for relevant documents
            gamma (float): Weight for irrelevant documents
            score_threshold (float): Minimum score threshold
            
        Returns:
            dict: Dictionary mapping query IDs to lists of retrieved document IDs
        """
        queries = self.parse_query(query_file)
        all_results = {}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("query_id,retrieved_docs\n") #First column: query_id, which is the last three digits in <number>…<number> tag in the query xml file. 
            
            for query_id, query_text in queries:
                print(f"Processing query {query_id}...")
                
                query_id, results = self.process_query(
                    query_id, query_text, use_feedback, feedback_iterations, 
                    alpha, beta, gamma, score_threshold
                )
                
                # Format results as space-separated document IDs
                doc_ids = [doc_id for doc_id, _ in results]
                f.write(f"{query_id},{' '.join(doc_ids)}\n")
                
                # Store results for evaluation
                all_results[query_id] = doc_ids
                
                print(f"Query {query_id} processed, retrieved {len(results)} documents")
                
        return all_results

    @staticmethod
    def evaluate_map(retrieved_docs, ground_truth):
        """
        Evaluate Mean Average Precision (MAP).
        
        MAP = mean(AP_1, AP_2, ..., AP_n) where AP is the average precision for a query.
        AP = sum(precision@k * rel(k)) / number_of_relevant_documents
        
        Args:
            retrieved_docs (dict): Dict mapping query_id to list of retrieved document IDs
            ground_truth (dict): Dict mapping query_id to list of relevant document IDs
        
        Returns:
            float: MAP score
        Comments:
            This function is used to test the performance of the model on the training data set (10 queries).
        """
        if not retrieved_docs or not ground_truth:
            return 0.0
        
        ap_scores = []
        
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in ground_truth:
                continue
                
            relevant = ground_truth[query_id]
            
            # If no relevant documents, skip this query
            if not relevant:
                continue
                
            # Calculate precision at each relevant document
            relevant_count = 0
            precision_sum = 0
            
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant:
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)
            
            # Average precision for this query
            if len(relevant) > 0:
                ap = precision_sum / len(relevant)
                ap_scores.append(ap)
        
        # Mean Average Precision across all queries
        if ap_scores:
            return sum(ap_scores) / len(ap_scores)
        else:
            return 0.0

    @staticmethod
    def load_ground_truth(gt_file):
        """
        Load ground truth from CSV file. (wm-2025-vector-space-mode/queries/ans_train.csv)
        
        Args:
            gt_file (str): Path to ground truth file
            
        Returns:
            dict: Dictionary mapping query IDs to lists of relevant document IDs
        """
        ground_truth = {}
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader)
            
            for row in reader:
                query_id = row[0]
                doc_ids = row[1].split() if len(row) > 1 else []
                ground_truth[query_id] = doc_ids
        
        return ground_truth


def grid_search(vsm, query_file, gt_file, output_base, param_grid):
    """
    Perform grid search over parameters to find optimal configuration.
    
    Args:
        vsm (VSM_BM25): VSM_BM25 instance
        query_file (str): Path to query file
        gt_file (str): Path to ground truth file
        output_base (str): Base name for output files
        param_grid (dict): Dictionary of parameter ranges
    
    Returns:
        tuple: (best_params, best_map) Best parameters and corresponding MAP score
    """
    # Load ground truth
    ground_truth = VSM_BM25.load_ground_truth(gt_file)
    
    best_map = 0
    best_params = {}
    results = []
    
    # Generate all parameter combinations
    num_combinations = (
        len(param_grid['iterations']) * 
        len(param_grid['alpha']) * 
        len(param_grid['beta']) * 
        len(param_grid['gamma']) * 
        len(param_grid['threshold']) *
        len(param_grid['k1']) *
        len(param_grid['b']) *
        len(param_grid['k3'])
    )
    
    print(f"Starting grid search with {num_combinations} parameter combinations...")
    
    # Not sure if there is a better way to do this, but I just use brute force to iterate through all combinations.
    # This is a bit ugly (nested for loops), but it works?
    # Should have used itertools.product to generate all combinations.
    count = 0
    for iterations in param_grid['iterations']:
        for alpha in param_grid['alpha']:
            for beta in param_grid['beta']:
                for gamma in param_grid['gamma']:
                    for threshold in param_grid['threshold']:
                        for k1 in param_grid['k1']:
                            for b in param_grid['b']:
                                for k3 in param_grid['k3']:
                                    count += 1
                                    print(f"Testing combination {count}/{num_combinations}")
                                    
                                    # Update BM25 parameters
                                    vsm.k1 = k1
                                    vsm.b = b
                                    vsm.k3 = k3
                                    
                                    params = {
                                        'iterations': iterations,
                                        'alpha': alpha,
                                        'beta': beta,
                                        'gamma': gamma,
                                        'threshold': threshold,
                                        'k1': k1,
                                        'b': b,
                                        'k3': k3
                                    }
                                    
                                    # Create output filename with parameters
                                    output_file = f"{output_base}_i{iterations}_a{alpha}_b{beta}_g{gamma}_t{threshold}_k1{k1}_b{b}_k3{k3}.csv"
                                    
                                    # Process queries with current parameters
                                    retrieved_docs = vsm.process_queries(
                                        query_file, output_file, True, 
                                        iterations, alpha, beta, gamma, threshold
                                    )
                                    
                                    # Evaluate MAP
                                    map_score = VSM_BM25.evaluate_map(retrieved_docs, ground_truth)
                                    
                                    results.append((params, map_score, output_file))
                                    
                                    print(f"MAP: {map_score:.4f} with parameters: {params}")
                                    
                                    # Update best parameters if better MAP found
                                    if map_score > best_map:
                                        best_map = map_score
                                        best_params = params
    
    # Sort results by MAP score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Write results to a summary file
    summary_file = f"{output_base}_summary.csv"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("iterations,alpha,beta,gamma,threshold,k1,b,k3,map,output_file\n")
        for params, map_score, output_file in results:
            f.write(f"{params['iterations']},{params['alpha']},{params['beta']},"
                   f"{params['gamma']},{params['threshold']},{params['k1']},"
                   f"{params['b']},{params['k3']},{map_score:.4f},{output_file}\n")
    
    print("\nGrid search complete!")
    print(f"Best MAP: {best_map:.4f}")
    print(f"Best parameters: {best_params}")
    print(f"Results summary written to: {summary_file}")
    
    return best_params, best_map


def main():
    """
    Main function to run the BM25 information retrieval system.
    
    Command line arguments:
        -r: Turn on relevance feedback
        -i: Input query file
        -o: Output ranked list file
        -m: Model directory
        -d: NTCIR directory
        -g: Ground truth file (for evaluation)
        --grid-search: Perform grid search over parameters
        --iterations: Number of feedback iterations
        --alpha: Alpha parameter for Rocchio
        --beta: Beta parameter for Rocchio
        --gamma: Gamma parameter for Rocchio
        --threshold: Score threshold
        --k1: BM25 k1 parameter
        --b: BM25 b parameter
        --k3: BM25 k3 parameter
    """
    parser = argparse.ArgumentParser(description='BM25 Information Retrieval System')
    parser.add_argument('-r', action='store_true', help='Turn on relevance feedback')
    parser.add_argument('-i', required=True, help='Input query file')
    parser.add_argument('-o', required=True, help='Output ranked list file')
    parser.add_argument('-m', required=True, help='Model directory')
    parser.add_argument('-d', required=True, help='NTCIR directory')
    parser.add_argument('-g', help='Ground truth file (for evaluation)')
    parser.add_argument('--grid-search', action='store_true', help='Perform grid search over parameters')
    parser.add_argument('--iterations', type=int, default=1, help='Number of feedback iterations')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for Rocchio')
    parser.add_argument('--beta', type=float, default=0.75, help='Beta parameter for Rocchio')
    parser.add_argument('--gamma', type=float, default=0, help='Gamma parameter for Rocchio')
    parser.add_argument('--threshold', type=float, default=550, help='Score threshold')
    parser.add_argument('--k1', type=float, default=1.2, help='BM25 k1 parameter')
    parser.add_argument('--b', type=float, default=0.75, help='BM25 b parameter')
    parser.add_argument('--k3', type=float, default=8, help='BM25 k3 parameter')
    parser.add_argument('--find-threshold', action='store_true', help='Find optimal threshold from ground truth')

    args = parser.parse_args()
    
    start_time = time.time()
        
    # Initialize VSM with BM25
    vsm = VSM_BM25(args.m, args.d)
    
    # Set BM25 parameters from command line
    vsm.k1 = args.k1
    vsm.b = args.b
    vsm.k3 = args.k3

    if args.grid_search:
        print("Performing grid search...")
        # Check if ground truth file is provided
        if not args.g:
            print("Error: Ground truth file (-g) required for grid search")
            return
        
        # Define parameter grid
        # These trials are just bullshit, the crucial point is to pre-process the three model files (inverted files, vocab, etc) to remove stop words....
        # and the threshold value is also a crucial point.
        param_grid = {
            'iterations': [1],#, 2],
            'alpha': [1], #0.8, 1.0],
            'beta': [0.75], #][0.5, 0.75, 0.85, 1],
            'gamma': [0], #0.1, 0.15],
            'threshold': [500,600,700],#[0.025,0.05, 0.1],
            'k1': [1.2],#[1.2, 1.5, 2.0],
            'b': [0.75],#[0.5, 0.75, 0.8],
            'k3': [800]#[0, 7, 8, 10]
        }
        
        # Perform grid search
        grid_search(vsm, args.i, args.g, args.o, param_grid)
    else:
        # Process queries with specified parameters
        retrieved_docs = vsm.process_queries(
            args.i, args.o, args.r, 
            args.iterations, args.alpha, args.beta, args.gamma, args.threshold
        )
        
        # Evaluate if ground truth is provided
        if args.g:
            ground_truth = VSM_BM25.load_ground_truth(args.g)
            map_score = VSM_BM25.evaluate_map(retrieved_docs, ground_truth)
            print(f"Mean Average Precision (MAP): {map_score:.4f}")
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()