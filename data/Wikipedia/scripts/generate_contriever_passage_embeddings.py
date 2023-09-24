# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import argparse
import pickle
import torch

from transformers import AutoTokenizer, AutoModel

import src.slurm
import src.contriever
import src.utils
import src.data
import src.normalize_text


def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in enumerate(passages):
            batch_ids.append(p["id"])
            if args.no_title or not "title" in p:
                text = p["text"]
            else:
                text = p["title"] + " " + p["text"]
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    print(f"Model loaded from {args.model_name_or_path}.", flush=True)
    model.eval()
    model = model.to(args.device)
    if not args.no_fp16:
        model = model.half()

    passages = src.data.load_passages(args.passages)

    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    print(f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.")

    allids, allembeddings = embed_passages(args, passages, model, tokenizer)

    save_file = os.path.join(args.output_dir, args.prefix + f"_{args.shard_id:02d}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(allids)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_path", type=str, default="./data/Wikipedia/wiki-pages/", help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, default="./data/Wikipedia/data/wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="prefix path to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_path", type=str, default="./facebook/contriever-msmarco", help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")

    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    src.slurm.init_distributed_mode(args)

    main(args)