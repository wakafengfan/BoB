# Copyright 2021 Haoyu Song
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
# -*- coding: utf-8 -*-


from argparse import ArgumentParser

from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer, load_vocab
from sklearn.model_selection import train_test_split

from configuration.config import *
from dataloader import read_convai2_split
from dataloader import read_ecdt2019_split
from dataloader import read_nli_split

dict_path = roberta_wwm_pt_path / "vocab.txt"

token_dict = load_vocab(
    dict_path=str(dict_path),
    # simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


"""
tokenizer:
input: list of texts
output: 
{
"input_ids": "", 
"token_type_ids": "", 
"attention_mask": ""
}

"""


def batch_tokenizer(texts, max_seq_len):
    dic = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }
    for i, text in enumerate(texts):
        token_ids, seg_ids = tokenizer.encode(text, maxlen=max_seq_len)
        att_masks = [1] * len(token_ids)

        dic["input_ids"].append(token_ids)
        dic["token_type_ids"].append(seg_ids)
        dic["attention_mask"].append(att_masks)

    dic["input_ids"] = sequence_padding(dic["input_ids"]).tolist()
    dic["token_type_ids"] = sequence_padding(dic["token_type_ids"]).tolist()
    dic["attention_mask"] = sequence_padding(dic["attention_mask"]).tolist()

    return dic


def preprocess(args):
    print(f"Reading {args.dataset_type} dataset...")
    train_persona, train_query, train_response = read_convai2_split(args.trainset) if args.dataset_type=='convai2' else read_ecdt2019_split(args.trainset)
    test_persona, test_query, test_response = read_convai2_split(args.testset) if args.dataset_type=='convai2' else read_ecdt2019_split(args.testset, split_type='test')

    positive_pre, positive_hyp = read_nli_split(args.nliset+'nli_positive.tsv')
    negative_pre, negative_hyp = read_nli_split(args.nliset+'nli_negative.tsv')

    assert len(train_persona) == len(train_query) == len(train_response)
    assert len(test_persona) == len(test_query) == len(test_response)

    train_persona, val_persona, train_query, val_query, train_response, val_response = train_test_split(
        train_persona,
        train_query,
        train_response,
        test_size=args.train_valid_split)
    print("Dataset loaded.")

    print("Tokenize...")
    # if args.dataset_type == 'convai2' and args.roberta:
    #     tokenizer = RobertaTokenizer.from_pretrained(args.encoder_model_name_or_path)
    # elif args.dataset_type == 'ecdt2019' and args.roberta:
    #     tokenizer = BertTokenizer.from_pretrained('./pretrained_models/bert/bert-base-chinese/')
    # else:
    #     tokenizer = BertTokenizer.from_pretrained(args.encoder_model_name_or_path)

    print("Tokenize persona...")
    train_persona_tokenized = batch_tokenizer(train_persona, args.max_source_length)
                                        # truncation=True,
                                        # padding=True,
                                        # max_length=args.max_source_length)
    train_persona_tokenized = {
            key: val
            for key, val in train_persona_tokenized.items()
        }
    val_persona_tokenized = batch_tokenizer(val_persona, args.max_source_length)
                                      # truncation=True,
                                      # padding=True,
                                      # max_length=args.max_source_length)
    val_persona_tokenized = {
            key: val
            for key, val in val_persona_tokenized.items()
        }
    test_persona_tokenized = batch_tokenizer(test_persona, args.max_source_length)
                                       # truncation=True,
                                       # padding=True,
                                       # max_length=args.max_source_length)
    test_persona_tokenized = {
            key: val
            for key, val in test_persona_tokenized.items()
        }

    print("Tokenize query...")
    train_query_tokenized = batch_tokenizer(train_query, args.max_source_length)
                                      # truncation=True,
                                      # padding=True,
                                      # max_length=args.max_source_length)
    train_query_tokenized = {
            key: val
            for key, val in train_query_tokenized.items()
        }
    val_query_tokenized = batch_tokenizer(val_query, args.max_source_length)
                                    # truncation=True,
                                    # padding=True,
                                    # max_length=args.max_source_length)
    val_query_tokenized = {
            key: val
            for key, val in val_query_tokenized.items()
        }
    test_query_tokenized = batch_tokenizer(test_query, args.max_source_length)
                                     # truncation=True,
                                     # padding=True,
                                     # max_length=args.max_source_length)
    test_query_tokenized = {
            key: val
            for key, val in test_query_tokenized.items()
        }

    print("Tokenize response...")
    train_response_tokenized = batch_tokenizer(train_response, args.max_target_length)
                                         # truncation=True,
                                         # padding=True,
                                         # max_length=args.max_target_length)
    train_response_tokenized = {
            key: val
            for key, val in train_response_tokenized.items()
        }
    val_response_tokenized = batch_tokenizer(val_response, args.max_target_length)
                                       # truncation=True,
                                       # padding=True,
                                       # max_length=args.max_target_length)
    val_response_tokenized = {
            key: val
            for key, val in val_response_tokenized.items()
        }
    test_response_tokenized = batch_tokenizer(test_response, args.max_target_length)
                                        # truncation=True,
                                        # padding=True,
                                        # max_length=args.max_target_length)
    test_response_tokenized = {
            key: val
            for key, val in test_response_tokenized.items()
        }

    print("Tokenize nli data...")
    positive_hyp_tokenized = batch_tokenizer(positive_hyp, args.max_target_length)
                                         # truncation=True,
                                         # padding=True,
                                         # max_length=args.max_target_length)
    positive_hyp_tokenized = {
        key: val
        for key, val in positive_hyp_tokenized.items()
    }
    positive_pre_tokenized = batch_tokenizer(positive_pre, args.max_source_length)
                                       # truncation=True,
                                       # padding=True,
                                       # max_length=args.max_source_length)
    positive_pre_tokenized = {
        key: val
        for key, val in positive_pre_tokenized.items()
    }

    negative_hyp_tokenized = batch_tokenizer(negative_hyp, args.max_target_length)
                                       # truncation=True,
                                       # padding=True,
                                       # max_length=args.max_target_length)
    negative_hyp_tokenized = {
        key: val
        for key, val in negative_hyp_tokenized.items()
    }
    negative_pre_tokenized = batch_tokenizer(negative_pre, args.max_source_length)
                                       # truncation=True,
                                       # padding=True,
                                       # max_length=args.max_source_length)
    negative_pre_tokenized = {
        key: val
        for key, val in negative_pre_tokenized.items()
    }
        
    if args.dataset_type=='convai2':
        path = './data/ConvAI2/convai2_tokenized/' if not args.roberta else './data/ConvAI2/convai2_roberta_tokenized/'
    else:
        path = './data/ECDT2019/ecdt2019_tokenized/'
    os.makedirs(path, exist_ok=True)
    print(f"Saving tokenized dict at {path}")
    
    with open(path+'train_persona.json','w') as train_persona:
        print("Dump train_persona")
        print(len(train_persona_tokenized['input_ids']))
        json.dump(train_persona_tokenized, train_persona)
    with open(path+'val_persona.json','w') as val_persona:
        print("Dump val_persona")
        print(len(val_persona_tokenized['input_ids']))
        json.dump(val_persona_tokenized, val_persona)
    with open(path+'test_persona.json','w') as test_persona:
        print("Dump test_persona")
        print(len(test_persona_tokenized['input_ids']))
        json.dump(test_persona_tokenized, test_persona)
    with open(path+'test_train_persona.json','w') as test_train_persona:
        print("Dump test_train_persona")
        test_train_persona_tokenized = {k:v[:10000] for k,v in train_persona_tokenized.items()}
        print(len(test_train_persona_tokenized['input_ids']))
        json.dump(test_train_persona_tokenized, test_train_persona)

    with open(path+'train_query.json','w') as train_query:
        print("Dump train_query")
        print(len(train_query_tokenized['input_ids']))
        json.dump(train_query_tokenized, train_query)
    with open(path+'val_query.json','w') as val_query:
        print("Dump val_query")
        print(len(val_query_tokenized['input_ids']))
        json.dump(val_query_tokenized, val_query)
    with open(path+'test_query.json','w') as test_query:
        print("Dump test_query")
        print(len(test_query_tokenized['input_ids']))
        json.dump(test_query_tokenized, test_query)
    with open(path+'test_train_query.json','w') as test_train_query:
        print("Dump test_train_query")
        test_train_query_tokenized = {k:v[:10000] for k,v in train_query_tokenized.items()}
        print(len(test_train_query_tokenized['input_ids']))
        json.dump(test_train_query_tokenized, test_train_query)

    with open(path+'train_response.json','w') as train_response:
        print("Dump train_response")
        print(len(train_response_tokenized['input_ids']))
        json.dump(train_response_tokenized, train_response)
    with open(path+'val_response.json','w') as val_response:
        print("Dump val_response")
        print(len(val_response_tokenized['input_ids']))
        json.dump(val_response_tokenized, val_response)
    with open(path+'test_response.json','w') as test_response:
        print("Dump test_response")
        print(len(test_response_tokenized['input_ids']))
        json.dump(test_response_tokenized, test_response)
    with open(path+'test_train_response.json','w') as test_train_response:
        print("Dump test_train_response")
        test_train_response_tokenized = {k:v[:10000] for k,v in train_response_tokenized.items()}
        print(len(test_train_response_tokenized['input_ids']))
        json.dump(test_train_response_tokenized, test_train_response)

    with open(path+'positive_pre.json','w') as positive_pre:
        print("Dump positive_pre")
        print(len(positive_pre_tokenized['input_ids']))
        json.dump(positive_pre_tokenized, positive_pre)
    with open(path+'positive_hyp.json','w') as positive_hyp:
        print("Dump positive_hyp")
        print(len(positive_hyp_tokenized['input_ids']))
        json.dump(positive_hyp_tokenized, positive_hyp)
    with open(path+'negative_pre.json','w') as negative_pre:
        print("Dump negative_pre")
        print(len(negative_pre_tokenized['input_ids']))
        json.dump(negative_pre_tokenized, negative_pre)
    with open(path+'negative_hyp.json','w') as negative_hyp:
        print("Dump negative_hyp")
        print(len(negative_hyp_tokenized['input_ids']))
        json.dump(negative_hyp_tokenized, negative_hyp)

if __name__ == "__main__":
    parser = ArgumentParser("Transformers EncoderDecoderModel Preprocessing")
    parser.add_argument(
        "--trainset",
        type=str,
        default=
        # "./data/ECDT2019/dialogues_train_samples.json")
        "./data/ECDT2019/dialogues_train.json")
        # "./data/ConvAI2/train_self_original_no_cands.txt")
    parser.add_argument(
        "--testset",
        type=str,
        default=
        "./data/ECDT2019/test_data_random.json")
        # "./data/ConvAI2/valid_self_original_no_cands.txt")
    parser.add_argument(
        "--nliset",
        type=str,
        default=
        "./data/ECDT2019/")
    
    parser.add_argument("--roberta", default=True, action="store_true")

    parser.add_argument("--train_valid_split", type=float, default=0.1)
    parser.add_argument("--max_source_length", type=int, default=64)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument(
        "--encoder_model_name_or_path",
        type=str,
        default=str(roberta_wwm_pt_path))

    parser.add_argument("--dataset_type",
                        type=str,
                        default='ecdt2019',
                        required=False)  # convai2, ecdt2019

    args = parser.parse_args()

    preprocess(args)
