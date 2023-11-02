#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import glob,time
import json
import pickle
import time
import zlib
from typing import List, Tuple, Dict, Iterator

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn

from scripts.utils.data_utils import RepTokenSelector
from scripts.data.qa_validation import calculate_matches, calculate_chunked_matches, calculate_matches_from_meta
from scripts.data.retriever_data import KiltCsvCtxSrc, TableChunk
from scripts.indexer.faiss_indexers import (
    DenseIndexer,
)
from scripts.models import init_biencoder_components
from scripts.models.biencoder import (
    BiEncoder,
    _select_span_with_token,
)
from scripts.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from scripts.utils.data_utils import Tensorizer
from scripts.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint


def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: str = None,
    selector: RepTokenSelector = None,
) -> T:
    n = len(questions)
    query_vectors = []
    # T means torch.Tensor
    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    batch_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token) for q in batch_questions
                    ]
                else:
                    batch_tensors = [tensorizer.text_to_tensor(" ".join([query_token, q])) for q in batch_questions]
            elif isinstance(batch_questions[0], T):
                batch_tensors = [q for q in batch_questions]
            else:
                batch_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]

            # TODO: this only works for Wav2vec pipeline but will crash the regular text pipeline
            # 下面这段话有问题
            # max_vector_len = max(q_t.size(1) for q_t in batch_tensors)
            # min_vector_len = min(q_t.size(1) for q_t in batch_tensors)
            max_vector_len = max(q_t.size(0) for q_t in batch_tensors)
            min_vector_len = min(q_t.size(0) for q_t in batch_tensors)

            if max_vector_len != min_vector_len:
                # TODO: _pad_to_len move to utils
                from scripts.models.reader import _pad_to_len
                batch_tensors = [_pad_to_len(q.squeeze(0), 0, max_vector_len) for q in batch_tensors]
            
            # 上面段话似乎有问题
            q_ids_batch = torch.stack(batch_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                print("Encoded queries {}".format(len(query_vectors)))

    query_tensor = torch.cat(query_vectors, dim=0)
    print("Total encoded queries tensor {}".format(query_tensor.size()))
    assert query_tensor.size(0) == len(questions)
    return query_tensor


class DenseRetriever(object):
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.selector = None

    def generate_question_vectors(self, questions: List[str], query_token: str = None) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            query_token=query_token,
            selector=self.selector,
        )


class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder;indexer is the DenseIndexer
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, batch_size, tensorizer)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        print("Data indexing completed.")

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        print("index search time: {} sec.".format(time.time() - time0))
        self.index = None
        return results


# works only with our distributed_faiss library
class DenseRPCRetriever(DenseRetriever):
    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index_cfg_path: str,
        dim: int,
        use_l2_conversion: bool = False,
        nprobe: int = 256,
    ):
        from distributed_faiss.client import IndexClient

        super().__init__(question_encoder, batch_size, tensorizer)
        self.dim = dim
        self.index_id = "dr"
        self.nprobe = nprobe
        print("Connecting to index server ...")
        self.index_client = IndexClient(index_cfg_path)
        self.use_l2_conversion = use_l2_conversion
        print("Connected")

    def load_index(self, index_id):
        from distributed_faiss.index_cfg import IndexCfg

        self.index_id = index_id
        print("Loading remote index {}".format(index_id))
        idx_cfg = IndexCfg()
        idx_cfg.nprobe = self.nprobe
        if self.use_l2_conversion:
            idx_cfg.metric = "l2"

        self.index_client.load_index(self.index_id, cfg=idx_cfg, force_reload=False)
        print("Index loaded")
        self._wait_index_ready(index_id)

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int = 1000,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        from distributed_faiss.index_cfg import IndexCfg

        buffer = []
        idx_cfg = IndexCfg()

        idx_cfg.dim = self.dim
        print("Index train num=%d", idx_cfg.train_num)
        idx_cfg.faiss_factory = "flat"
        index_id = self.index_id
        self.index_client.create_index(index_id, idx_cfg)

        def send_buf_data(buf, index_client):
            buffer_vectors = [np.reshape(encoded_item[1], (1, -1)) for encoded_item in buf]
            buffer_vectors = np.concatenate(buffer_vectors, axis=0)
            meta = [encoded_item[0] for encoded_item in buf]
            index_client.add_index_data(index_id, buffer_vectors, meta)

        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                send_buf_data(buffer, self.index_client)
                buffer = []
        if buffer:
            send_buf_data(buffer, self.index_client)
        print("Embeddings sent.")
        self._wait_index_ready(index_id)

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100, search_batch: int = 512
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :param search_batch:
        :return:
        """
        if self.use_l2_conversion:
            aux_dim = np.zeros(len(query_vectors), dtype="float32")
            query_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
            print("query_hnsw_vectors {}".format(query_vectors.shape))
            self.index_client.cfg.metric = "l2"

        results = []
        for i in range(0, query_vectors.shape[0], search_batch):
            time0 = time.time()
            query_batch = query_vectors[i : i + search_batch]
            print("query_batch: %s", query_batch.shape)
            # scores, meta = self.index_client.search(query_batch, top_docs, self.index_id)

            scores, meta = self.index_client.search_with_filter(
                query_batch, top_docs, self.index_id, filter_pos=3, filter_value=True
            )

            print("index search time: %f sec.", time.time() - time0)
            results.extend([(meta[q], scores[q]) for q in range(len(scores))])
        return results

    def _wait_index_ready(self, index_id: str):
        from distributed_faiss.index_state import IndexState
        # TODO: move this method into IndexClient class
        while self.index_client.get_state(index_id) != IndexState.TRAINED:
            print("Remote Index is not ready ...")
            time.sleep(60)
        print(
            "Remote Index is ready. Index data size {}".format(self.index_client.get_ntotal(index_id)),
        )


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    print("validating passages. size={}".format(len(passages)))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits {}".format(top_k_hits))
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    print("Validation results: top k documents hits accuracy {}".format(top_k_hits))
    return match_stats.questions_doc_hits


def validate_from_meta(
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
    meta_compressed: bool,
) -> List[List[bool]]:

    match_stats = calculate_matches_from_meta(
        answers, result_ctx_ids, workers_num, match_type, use_title=True, meta_compressed=meta_compressed
    )
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits {}".format(top_k_hits))
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    print("Validation results: top k documents hits accuracy {}".format(top_k_hits))
    return match_stats.questions_doc_hits


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i] # retrieval results scores
        hits = per_question_hits[i] # validate answers in retrived questions or not
        docs = [passages[doc_id] for doc_id in results_and_scores[0]] # BiEncoderPassage
        scores = [str(score) for score in results_and_scores[1]] # retrieval scores for every ctx passage
        ctxs_num = len(hits)

        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": results_and_scores[0][c],
                    "title": docs[c][1],
                    "text": docs[c][0],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }

        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4,ensure_ascii=False) + "\n")
    print("Saved results * scores  to {}".format(out_file))


# TODO: unify with save_results
def save_results_from_meta(
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    rpc_meta_compressed: bool = False,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [doc for doc in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": docs[c][0],
                    "title": zlib.decompress(docs[c][2]).decode() if rpc_meta_compressed else docs[c][2],
                    "text": zlib.decompress(docs[c][1]).decode() if rpc_meta_compressed else docs[c][1],
                    "is_wiki": docs[c][3],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }
        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    print("Saved results * scores  to {}".format(out_file))


def iterate_encoded_files(vector_files: list, path_id_prefixes: List = None) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        print("Reading file {}".format(file))
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc


def validate_tables(
    passages: Dict[object, TableChunk],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_chunked_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_chunk_hits = match_stats.top_k_chunk_hits
    top_k_table_hits = match_stats.top_k_table_hits

    print("Validation results: top k documents hits {}".format(top_k_chunk_hits))
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_chunk_hits]
    print("Validation results: top k table chunk hits accuracy {}".format(top_k_hits))

    print("Validation results: top k tables hits {}".format(top_k_table_hits))
    top_k_table_hits = [v / len(result_ctx_ids) for v in top_k_table_hits]
    print("Validation results: top k tables accuracy {}".format(top_k_table_hits))

    return match_stats.top_k_chunk_hits


def get_all_passages(ctx_sources):
    all_passages = {} # ctx_sources: dpr.data.retriever_data.CsvCtxSrc
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)
        print("Loaded ctx data: {}".format(len(all_passages)))

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages


@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)

    # load dpr model
    saved_state = load_states_from_checkpoint(cfg.model_file)

    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    print("CFG (after gpu  configuration):")
    print("{}".format(OmegaConf.to_yaml(cfg)))
    
    # load encoder(BERT) model
    cfg.encoder.pretrained_model_cfg="/data/yangjun/fact/wikifact/bert-base-uncased/"
    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    print("Loading saved model state ...")
    encoder.load_state(saved_state, strict=False)

    encoder_path = cfg.encoder_path
    if encoder_path:
        # ctx_model: for context encoder
        print("Selecting encoder: {}".format(encoder_path))
        encoder = getattr(encoder, encoder_path)
    else:
        # question_model: for question encoder
        print("Selecting standard question encoder")
        encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    encoder.eval()

    model_to_load = get_model_obj(encoder)
    vector_size = model_to_load.get_out_size()
    print("Encoder vector_size={}".format(vector_size))

    # get claim and searched documents
    claims = []
    documents = []
    
    if not cfg.claim_dataset:
        print("Please specify claim_dataset to use")
        return
    
    # load claim test dataset
    ds_key = cfg.claim_dataset  # eg: fever_test
    print("claim_dataset: {}".format(ds_key))

    fc_src = hydra.utils.instantiate(cfg.datasets[ds_key])   # map fever_test to scripts.data.retriever_data.JsonCtxSrc
    fc_src.load_data()

    total_claims = len(fc_src)
    for i in range(total_claims):
        fc_sample = fc_src[i]
        claim, document = fc_sample.claim, fc_sample.document
        claims.append(claim)
        documents.append(document)
    
    print("claim num: {}, document num: {}".format(len(claims), len(documents)))

    # 看不懂，似乎也不影响
    if cfg.rpc_retriever_cfg_file:
        index_buffer_sz = 1000 # default is None
        retriever = DenseRPCRetriever(
            encoder,
            cfg.batch_size,
            tensorizer,
            cfg.rpc_retriever_cfg_file,
            vector_size,
            use_l2_conversion=cfg.use_l2_conversion,
        )
    else:
        index = hydra.utils.instantiate(cfg.indexers[cfg.indexer]) # default indexer is flat;
        print("Local Index class {}".format(type(index)))
        index_buffer_sz = index.buffer_size # 50000
        index.init_index(vector_size) # vector_size:768
        retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index)

    claims_tensor = retriever.generate_question_vectors(claims)
    if fc_src.selector:
        print("Using custom representation token selector")
        retriever.selector = fc_src.selector
    
    # if there is no index at the specific location, the index will be created from encoded_ctx_files
    index_path = cfg.index_path
    if cfg.rpc_retriever_cfg_file and cfg.rpc_index_id:
        # False
        retriever.load_index(cfg.rpc_index_id)
    elif index_path and index.index_exists(index_path):
        # None
        print("Index path: {}".format(index_path))
        retriever.index.deserialize(index_path)
    else:
        # send data for indexing
        id_prefixes = []
        ctx_sources = []
        for ctx_src in cfg.ctx_datasets:
            ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
            id_prefixes.append(ctx_src.id_prefix)
            ctx_sources.append(ctx_src)
            print("ctx_sources: {}".format(type(ctx_src)))

        print("id_prefixes per dataset: {}".format(id_prefixes))

        # index all passages
        ctx_files_patterns = cfg.encoded_ctx_files

        print("ctx_files_patterns: {}".format(ctx_files_patterns))
        if ctx_files_patterns:
            assert len(ctx_files_patterns) == len(id_prefixes), "ctx len={} pref leb={}".format(
                len(ctx_files_patterns), len(id_prefixes)
            )
        else:
            assert (
                index_path or cfg.rpc_index_id
            ), "Either encoded_ctx_files or index_path pr rpc_index_id parameter should be set."

        input_paths = []
        path_id_prefixes = []
        for i, pattern in enumerate(ctx_files_patterns):
            pattern_files = glob.glob(pattern)
            pattern_id_prefix = id_prefixes[i]
            input_paths.extend(pattern_files)
            path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
        print("Embeddings files id prefixes: {}".format(path_id_prefixes))
        print("Reading all passages data from files: {}".format(input_paths))
        retriever.index_encoded_data(input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes)
        if index_path:
            retriever.index.serialize(index_path) # default is None;
        
    # get top k results
    starttime=time.time()
    top_results_and_scores = retriever.get_top_docs(claims_tensor.numpy(), cfg.n_docs) # default is 100
    endtime=time.time()
    
    if cfg.use_rpc_meta:
        claims_doc_hits = validate_from_meta(
            documents,
            top_results_and_scores,
            cfg.validation_workers,
            cfg.match,
            cfg.rpc_meta_compressed,
        )
        if cfg.out_file:
            save_results_from_meta(
                claims,
                documents,
                top_results_and_scores,
                claims_doc_hits,
                cfg.out_file,
                cfg.rpc_meta_compressed,
            )
    else:
        all_passages = get_all_passages(ctx_sources)
        if cfg.validate_as_tables:
            # deafault is False
            claims_doc_hits = validate_tables(
                all_passages,
                documents,
                top_results_and_scores,
                cfg.validation_workers,
                cfg.match,
            )

        else:
            claims_doc_hits = validate(
                all_passages,
                documents,
                top_results_and_scores,
                cfg.validation_workers,
                cfg.match,
            )

        if cfg.out_file:
            save_results(
                all_passages,
                claims,
                documents,
                top_results_and_scores,
                claims_doc_hits,
                cfg.out_file,
            )
        print("querys:{},contexts:{},total_time:{},ave_time:{}".format(len(claims),len(all_passages),round(endtime-starttime,3),round(endtime-starttime)/len(claims),3))
    
    if cfg.kilt_out_file:
        kilt_ctx = next(iter([ctx for ctx in ctx_sources if isinstance(ctx, KiltCsvCtxSrc)]), None)
        if not kilt_ctx:
            raise RuntimeError("No Kilt compatible context file provided")
        assert hasattr(cfg, "kilt_out_file")
        kilt_ctx.convert_to_kilt(fc_src.kilt_gold_file, cfg.out_file, cfg.kilt_out_file)


if __name__ == "__main__":
    main()
