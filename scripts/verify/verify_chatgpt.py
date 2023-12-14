import openai
import torch
import json
import argparse
import random
import re
import sys
from tqdm import tqdm
from fever.scorer import fever_score
from prettytable import PrettyTable

from utils import log

FEVER_PROMPT = """Verify the claim according to the evidence titles and sentences.Several examples are given as follows.

claim: Roman Atwood is a content creator.
evidence: <title> Roman Atwood <sentence> He is best known for his vlogs , where he posts updates about his life on a daily basis .
label: SUPPORTS

claim: Adrienne Bailon is an accountant.
evidence: <title> Adrienne Bailon <sentence> Adrienne Eliza Houghton ( Bailon ; born October 24 , 1983 ) is an American singer-songwriter , recording artist , actress , dancer and television personality .
label: REFUTES

claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
evidence: <title> Nikolaj Coster-Waldau <sentence> He then played Detective John Amsterdam in the short-lived Fox television series New Amsterdam ( 2008 ) , as well as appearing as Frank Pike in the 2009 Fox television film Virtuality , originally intended as a pilot . <title> Fox Broadcasting Company <sentence> The Fox Broadcasting Company ( often shortened to Fox and stylized as FOX ) is an American English language commercial broadcast television network that is owned by the Fox Entertainment Group subsidiary of 21st Century Fox .
label: SUPPORTS

claim: Peggy Sue Got Married is a Egyptian film released in 1986.
evidence: <title> Peggy Sue Got Married <sentence> Peggy Sue Got Married is a 1986 American comedy-drama film directed by Francis Ford Coppola starring Kathleen Turner as a woman on the verge of a divorce , who finds herself transported back to the days of her senior year in high school in 1960 . <title> Francis Ford Coppola <sentence> Francis Ford Coppola ( born April 7 , 1939 ) , also credited as Francis Coppola , is a semi-retired American film director , producer , and screenwriter .
label: REFUTES

claim: System of a Down briefly disbanded in limbo.
evidence: <title> System of a Down (album) <sentence> System of a Down is the debut studio album by Armenian American metal band System of a Down , released on June 30 , 1998 , by American Recordings and Columbia Records . <title> Limbo (video game) <sentence> Limbo is a puzzle platform video game developed by independent studio Playdead . <title> System of a Down <sentence> System of a Down , sometimes shortened to SOAD or System , is an heavy metal band from Glendale , California , formed in 1994 . <title> Limbo (video game) <sentence> The game was released in July 2010 on Xbox Live Arcade , and has since been ported to several other systems , including the PlayStation 3 and Microsoft Windows . <title> Limbo (video game) <sentence> Limbo received positive reviews , but its minimal story polarised critics ; some critics found the open ended work to have deeper meaning that tied well with the game 's mechanics , while others believed the lack of significant plot and abrupt ending detracted from the game .
label: NOT ENOUGH INFO

claim: Beautiful reached number two on the Billboard Hot 100 in 2003.
evidence: <title> Beautiful (Christina_Aguilera_song) <sentence> `` Beautiful '' is a song recorded by American singer Christina Aguilera for her fourth studio album , Stripped ( 2002 ) . The song peaked at number two on the Billboard Hot 100 in the United States , where it was certified Gold for 500,000 units shipped . It was released as the album 's second single on November 16 , 2002 . It won a Grammy Award for Best Female Pop Vocal Performance and was also nominated for Song of the Year at the 2004 ceremony . A pop and RB ballad , `` Beautiful '' was written and produced by Linda Perry .
label: NOT ENOUGH INFO

claim: [[CLAIM]]
evidence: [[EVIDENCE]]
label: 
"""

label_map = {
    'true': 0, 'support': 0, 'supports': 0, 'yes': 0,
    "it's impossible to say": 1, 'not enough info': 1, 'not enough information': 1, 'nei': 1,
    'false': 2, 'refute': 2, 'refutes': 2, 'no': 2}

def llm(prompt, stop=["\n"]):
    response = openai.ChatCompletion.create(
            engine="ChatGPT",
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
            messages=[{"role":"user","content":prompt}]
        )
    return response.choices[0].message.content

def process_sent(sentence):
    sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
    sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
    sentence = re.sub(" -LRB-", " ( ", sentence)
    sentence = re.sub("-RRB-", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence

def process_wiki_title(title):
    title = re.sub("_", " ", title)
    title = re.sub(" -LRB-", " ( ", title)
    title = re.sub("-RRB-", " )", title)
    title = re.sub("-COLON-", ":", title)
    return title

def convert_evidence(evidence_list):
    text = ""
    pred_list = []
    for evidence in  evidence_list:
        pred_list.append([evidence[0], evidence[1]])
        if evidence[3] > 0:
            text += "<title> {} <sentence> {} ".format(process_wiki_title(evidence[0]), process_sent(evidence[2]))
    return text, pred_list

def verify(dataset, logger):
    pred_results = []
    save_results = {}
    for data in tqdm(dataset):
        claim = data['claim']
        evidence, pred_evidence = convert_evidence(data['evidence'])
        text = FEVER_PROMPT.replace("[[CLAIM]]", claim)
        prompt = text.replace("[[EVIDENCE]]", evidence)

        try:
            output = llm(prompt)

            pred = output.lower().strip()
            
            if pred in label_map:
                pred = label_map[pred]
            else:
                logger.info("Alert! prediction error id:{}, prediction: {}".format(data['claim'], pred))
                pred = random.sample([0, 1, 2], 1)[0]
        except:
            pred = random.sample([0, 1, 2], 1)[0]
            
        if pred == 2:
            pred_label = 'REFUTES'
        elif pred == 0:
            pred_label = 'SUPPORTS'
        elif pred == 1:
            pred_label = 'NOT ENOUGH INFO'
        
        print("id: {}, claim: '{}', pred_label: '{}', pred_evidence: {}". format(data['id'], data['claim'], pred_label, pred_evidence))
        pred_results.append({
            'id': data['id'],
            'pred_label': pred_label,
            'pred_evidence': pred_evidence,
        })
        save_results[data['id']] = {
            'id': data['id'],
            'pred_label': pred_label,
            'pred_evidence': pred_evidence,
        }
    return pred_results, save_results
    
def evaluate(predictions, gold_data_path):
    predicted_labels =[]
    predicted_evidence = []
    actual = []
    ids = dict()
    
    for pred in predictions:
        ids[pred["id"]] = len(predicted_labels)
        predicted_labels.append(pred["pred_label"])
        predicted_evidence.append(pred["pred_evidence"])
        actual.append(0)

    with open(gold_data_path, "r") as actual_file:
        for line in actual_file:
            actual[ids[json.loads(line)["id"]]] = json.loads(line)

    predictions = []
    for ev,label in zip(predicted_evidence, predicted_labels):
        predictions.append({"predicted_evidence":ev,"predicted_label":label})

    score,acc,precision,recall,f1 = fever_score(predictions,actual)

    tab = PrettyTable()
    tab.field_names = ["FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
    tab.add_row((round(score,4),round(acc,4),round(precision,4),round(recall,4),round(f1,4)))

    print(tab)

def main(args):
    log_path = args.log_path + args.dataset_name + '_verify_with_chatgpt.log'
    tmp_log_path = args.log_path + args.dataset_name + '_verify_with_chatgpt_tmp.log'
    sys.stdout = open(tmp_log_path, mode='w')

    # init logger
    logger = log.get_logger(log_path)
    logger.info(args)

    # load data
    logger.info("loading data......")
    # data_path = args.data_path + args.dataset_name + "/processed/" + args.mode + ".json"
    dataset = []
    with open(args.data_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    dataset = dataset[:1000]
    print(len(dataset))
    
    # load model
    logger.info("loading model......")
    pred_results, save_results = verify(dataset, logger)

    tmp_save_path = args.save_path + args.dataset_name + "_test_verify_with_chatgpt_tmp.json"
    with open(tmp_save_path, 'w') as f:
        json.dump(pred_results, f, indent=2, ensure_ascii=False)

    # save outputs
    outputs = []
    count = 0
    with open(args.gold_data_path, 'r') as f:
        for line in f:
            if count >= 1000:
                break
            inst = json.loads(line)
            id = inst['id']
            data = save_results[id]
            outputs.append({
                'id': id,
                'claim': inst['claim'],
                'gold_evidence': inst['evidence'],
                'gold_label': inst['label'],
                'pred_evidence': data['pred_evidence'],
                'pred_label': data['pred_label']
            })
            count += 1
            
    save_path = args.save_path + args.dataset_name + "_test_verify_with_chatgpt.json"
    with open(save_path, 'w') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    logger.info("Finished!")

    # convert predictions and gold for calculating fever score
    evaluate(pred_results, args.gold_data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--log_path', type=str, default='./logs/')
    # parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--data_path', type=str, default='./data/FEVER/SelectData/bert_eval.json')
    parser.add_argument('--gold_data_path', type=str, default='./data/FEVER/SelectData/dev_eval.json')
    parser.add_argument('--dataset_name', type=str, default='FEVER', choices=['SCIFACT'])
    parser.add_argument('--save_path', type=str, default='./results/ablation/')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'dev', 'test'])

    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda:0")

    # wikipedia arguments
    parser.add_argument('--db_path', type=str, default='./data/Wikipedia/data/wikipedia.db')
    parser.add_argument('--db_table', type=str, default='documents')

    # model
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=512)
    # parser.add_argument('--model_name', type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    # parser.add_argument('--cache_dir', type=str, default='./MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')
    parser.add_argument('--model_name', type=str, default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument('--cache_dir', type=str, default='./MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
    parser.add_argument('--checkpoint', type=str, default='./models/deberta_verify2.pth')
    
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(int(args.gpu))
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    # chatgpt api
    openai.api_type = 'azure'
    openai.api_base = 'https://1027gpt.openai.azure.com/'
    openai.api_version = '2023-03-15-preview'
    openai.api_key = '2d568b2bb17b4e02a2b424783e313176'
    
    main(args)
