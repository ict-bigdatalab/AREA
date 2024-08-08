import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np

import copy


import logging, argparse, os, time
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import (BertConfig, BertTokenizer, AdamW,
                          get_linear_schedule_with_warmup)
from transformers import BertTokenizer

from myutils.semantic_helper import SemanticHelper
from utils import read_qds_pairs
from modeling import RankingBERT_Train
from marcodoc.dataset import MSMARCODataset_file_qd, get_collate_function, CollectionDataset
from myutils.word_recover.Bert_word_recover import BERTWordRecover
from myutils.attacker.attacker import Attacker

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%d %H:%M:%S',
                    level=logging.INFO)


def read_resources(embed_path, embed_cos_matrix_path, embed_name,
                   attack_qd_path, black_model_ranked_list_score_path, bert_tokenizer,
                   bert_vocab_path, max_query_length, max_doc_length, collection_memmap_dir):
    '''
    :return:
    '''
    synonym_helper = SemanticHelper(embed_path, embed_cos_matrix_path)
    synonym_helper.build_vocab()
    synonym_helper.load_embedding_cos_sim_matrix()

    word_re = BERTWordRecover(embed_name, bert_tokenizer, bert_vocab_path, max_query_length, max_doc_length)

    attack_qds = read_qds_pairs(attack_qd_path)

    ori_ranked_list_qds = read_qds_pairs(black_model_ranked_list_score_path)

    collection = CollectionDataset(collection_memmap_dir)

    return synonym_helper, word_re, attack_qds, ori_ranked_list_qds, collection

class MultiViewRepresentationLearning:
    def __init__(self, surrogate_model, n_clusters, device):
        self.surrogate_model = surrogate_model
        self.n_clusters = n_clusters
        self.device = device

    def derive_viewers(self, document_embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(document_embeddings)
        viewers = kmeans.cluster_centers_
        return viewers

    def generate_multi_view_representations(self, target_embedding, viewers):
        fc_layer = nn.Linear(target_embedding.size(1), target_embedding.size(1)).to(self.device)
        relu = nn.ReLU()
        optimizer = optim.Adam(fc_layer.parameters(), lr=0.001)
        
        target_embedding = target_embedding.unsqueeze(0)
        viewers = torch.tensor(viewers).to(self.device)
        
        for epoch in range(100):  # Number of epochs can be adjusted
            optimizer.zero_grad()
            multi_view_reps = relu(fc_layer(target_embedding))
            loss_squ = ((multi_view_reps - viewers) ** 2).sum() + ((multi_view_reps - target_embedding) ** 2).sum()
            loss_cos = -torch.sum(torch.cosine_similarity(multi_view_reps, viewers, dim=1))
            loss = loss_squ + 0.1 * loss_cos  # lambda is set to 0.1, can be adjusted
            loss.backward()
            optimizer.step()
        
        return multi_view_reps.detach().cpu().numpy()

    def obtain_counter_viewers(self, target_embedding, corpus_embeddings, initial_set_indices, n_counter_viewers):
        distances = torch.cdist(target_embedding.unsqueeze(0), corpus_embeddings, p=2).squeeze(0)
        sorted_indices = torch.argsort(distances)
        counter_viewers = []
        for idx in sorted_indices:
            if idx not in initial_set_indices:
                counter_viewers.append(corpus_embeddings[idx].cpu().numpy())
            if len(counter_viewers) >= n_counter_viewers:
                break
        return np.array(counter_viewers)


class ViewWiseContrastiveAttack:
    def __init__(self, surrogate_model, device, temperature=0.07):
        self.surrogate_model = surrogate_model
        self.device = device
        self.temperature = temperature

    def view_wise_contrastive_loss(self, multi_view_reps, viewers, counter_viewers):
        loss = 0
        for i in range(len(multi_view_reps)):
            numerator = torch.exp(torch.dot(multi_view_reps[i], viewers[i]) / self.temperature)
            denominator = numerator + torch.sum(torch.exp(torch.matmul(multi_view_reps[i], counter_viewers.T) / self.temperature))
            loss -= torch.log(numerator / denominator)
        return loss

    def perturbation_word_selection(self, target_document, multi_view_reps, viewers, counter_viewers):
        target_document.requires_grad = True
        loss = self.view_wise_contrastive_loss(multi_view_reps, viewers, counter_viewers)
        loss.backward()
        gradients = target_document.grad
        word_importance = torch.norm(gradients, dim=1)
        top_m_indices = torch.argsort(word_importance, descending=True)[:10]  # Select top 10 words
        return top_m_indices

    def embedding_perturbation_and_synonym_substitution(self, target_document, top_m_indices, synonym_dict, num_iterations=10, epsilon=0.1):
        perturbed_document = target_document.clone().detach()
        for _ in range(num_iterations):
            perturbed_document.requires_grad = True
            loss = self.view_wise_contrastive_loss(multi_view_reps, viewers, counter_viewers)
            loss.backward()
            with torch.no_grad():
                gradients = perturbed_document.grad
                perturbations = epsilon * gradients / torch.norm(gradients, dim=1, keepdim=True)
                perturbed_document[top_m_indices] += perturbations[top_m_indices]
                perturbed_document = torch.clamp(perturbed_document, -1, 1)  # Ensure embeddings are within a valid range

        # Synonym substitution
        for idx in top_m_indices:
            original_word_embedding = perturbed_document[idx].detach().cpu().numpy()
            synonyms = synonym_dict.get(idx.item(), [])
            best_synonym = None
            best_similarity = -float('inf')
            for synonym in synonyms:
                synonym_embedding = synonym['embedding']
                similarity = np.dot(original_word_embedding, synonym_embedding) / (np.linalg.norm(original_word_embedding) * np.linalg.norm(synonym_embedding))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_synonym = synonym['word']
            if best_synonym:
                perturbed_document[idx] = torch.tensor(synonym['embedding']).to(self.device)

        return perturbed_document

def main():
    args = run_parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    doc_list_length = args.doc_list_length
    assert doc_list_length % args.batch_size == 0
    read_num = int(doc_list_length / args.batch_size)

    # Load models
    ori_model_path = args.ori_model_path
    ori_config = BertConfig.from_pretrained(ori_model_path)
    ori_model = RankingBERT_Train.from_pretrained(ori_model_path, config=ori_config)
    ori_model.to(args.device)
    if args.n_gpu > 1:
        ori_model = torch.nn.DataParallel(ori_model)

    surr_model_path = args.surrogate_model_path
    surr_config = BertConfig.from_pretrained(surr_model_path)
    surr_model = RankingBERT_Train.from_pretrained(surr_model_path, config=surr_config)
    surr_model.to(args.device)
    if args.n_gpu > 1:
        surr_model = torch.nn.DataParallel(surr_model)
    surr_model_state_dict = surr_model.state_dict()
    surr_model_state_dict = copy.deepcopy(surr_model_state_dict)

    for name, param in surr_model.named_parameters():
        args.embed_name = name
        break
    print(args.embed_name)

    logger.info("evaluation parameters %s", args)

    # Create global resources
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    synonym_helper, word_recover, attack_qds, ori_qds, collection = read_resources(
        args.embed_path, args.embed_cos_matrix_path, args.embed_name,
        args.attack_qd_path, args.black_model_ranked_list_score_path, bert_tokenizer, args.bert_vocab_path,
        args.max_query_length, args.max_doc_length, args.collection_memmap_dir)

    max_attack_word_number = args.max_attack_word_number
    attack_save_path = args.save_doc_tokens_path
    attacked_docs_dict = {}
    attacked_docs_score_dict = {}

    # Create dataset
    mode = 'dev'
    dev_dataset = MSMARCODataset_file_qd(mode, args.black_model_ranked_list_path,
                                         args.collection_memmap_dir, args.tokenize_dir,
                                         args.bert_tokenizer_path,
                                         args.max_query_length, args.max_doc_length)
    collate_fn = get_collate_function(mode=mode)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                num_workers=args.data_num_workers,
                                collate_fn=collate_fn)
    dataloader_iter = enumerate(dev_dataloader)

    save_attacked_docs_f = open(attack_save_path, 'a')
    previous_done = args.previous_done
    for j in range(previous_done * read_num):
        dataloader_iter.__next__()
    qid_list_t = tqdm(list(attack_qds.keys())[previous_done:])

    tested_q_num = 0
    for qid in qid_list_t:
        tested_q_num += 1

        attack_docid_list = list(attack_qds[qid].keys())
        batch_list = []
        docid_list = []
        for i in range(read_num):
            batch_index, data = dataloader_iter.__next__()
            batch_list.append(batch)
            docid_list.append(docids)

        attack_docid_list_t = tqdm(attack_docid_list)
        for attack_docid in attack_docid_list_t:
            surr_model.load_state_dict(surr_model_state_dict)
            ori_score = attack_qds[qid][attack_docid]


            initial_set_embeddings = []
            for batch in batch_list:
                with torch.no_grad():
                    outputs = surr_model(**batch)
                    initial_set_embeddings.append(outputs.last_hidden_state.mean(dim=1))
            initial_set_embeddings = torch.cat(initial_set_embeddings, dim=0)


            mvrl = MultiViewRepresentationLearning(surr_model, n_clusters=5, device=args.device)
            viewers = mvrl.derive_viewers(initial_set_embeddings.cpu().numpy())
            target_embedding = initial_set_embeddings[docid_list[0].index(attack_docid)]
            multi_view_reps = mvrl.generate_multi_view_representations(target_embedding, viewers)


            corpus_embeddings = initial_set_embeddings  
            initial_set_indices = [docid for sublist in docid_list for docid in sublist]
            counter_viewers = mvrl.obtain_counter_viewers(target_embedding, corpus_embeddings, initial_set_indices, n_counter_viewers=5)


            vwca = ViewWiseContrastiveAttack(surr_model, device=args.device)
            top_m_indices = vwca.perturbation_word_selection(target_embedding, multi_view_reps, viewers, counter_viewers)

            synonym_dict = {}  
            perturbed_document = vwca.embedding_perturbation_and_synonym_substitution(target_embedding, top_m_indices, synonym_dict)


            new_doc_token_id_list = perturbed_document.argmax(dim=1).tolist()
            new_doc_token_id_list = [idx for idx in new_doc_token_id_list if idx != 0] 
            with torch.no_grad():
                new_doc_input = {
                    'input_ids': torch.tensor(new_doc_token_id_list).unsqueeze(0).to(args.device),
                    'token_type_ids': torch.zeros_like(torch.tensor(new_doc_token_id_list)).unsqueeze(0).to(args.device),
                    'attention_mask': torch.ones_like(torch.tensor(new_doc_token_id_list)).unsqueeze(0).to(args.device)
                }
                new_score = ori_model(**new_doc_input).logits.item()

            attack_doc_key = str(qid) + '_' + str(attack_docid)
            attacked_docs_dict[attack_doc_key] = new_doc_token_id_list
            attacked_docs_score_dict[attack_doc_key] = new_score

        for qid_docid in attacked_docs_dict:
            attacked_doc = word_recover.recover_doc(qid_docid.split('_')[1], attacked_docs_dict[qid_docid], collection, args.max_doc_length)
            to_write = qid_docid + '\t' + attacked_doc + '\t' + str(attacked_docs_score_dict[qid_docid])
            save_attacked_docs_f.write(to_write + '\n')
        attacked_docs_dict = {}

if __name__ == "__main__":
    main()