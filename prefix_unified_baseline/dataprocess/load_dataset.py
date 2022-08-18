import json
import random
import re
from openprompt.data_utils import InputExample
import pandas as pd

random.seed(10086)
include = ['boolq', 'commonsense', 'squad1.1', ]


def load_boolq():
    with open('../../datasets/boolq/train.jsonl', 'r', encoding='utf-8') as f_in:
        f_out = open('../../datasets/boolq/train_unified.jsonl', 'w', encoding='utf-8')
        for line in f_in.readlines():
            json_obj = json.loads(line)
            context = json_obj['passage']
            question = json_obj['question'] + '?'
            answer = str(json_obj['answer'])
            json_out_obj = json.dumps({'context': context, 'question': question, 'answer': answer}, ensure_ascii=False)
            f_out.write(json_out_obj + '\n')
        f_out.close()
    with open('../../datasets/boolq/dev.jsonl', 'r', encoding='utf-8') as f_in:
        dev_out_file = open('../../datasets/boolq/dev_unified.jsonl', 'w', encoding='utf-8')
        test_out_file = open('../../datasets/boolq/test_unified.jsonl', 'w', encoding='utf-8')
        dev_cnt = 0
        test_cnt = 0
        for line in f_in.readlines():
            json_obj = json.loads(line)
            ran_score = random.randint(1, 100)
            context = json_obj['passage']
            question = json_obj['question'] + '?'
            answer = str(json_obj['answer'])
            json_out_obj = json.dumps({'context': context, 'question': question, 'answer': answer}, ensure_ascii=False)
            if ran_score > 30:
                dev_out_file.write(json_out_obj + '\n')
                dev_cnt += 1
            else:
                test_out_file.write(json_out_obj + '\n')
                test_cnt += 1
    dev_out_file.close()
    test_out_file.close()
    print(dev_cnt)
    print(test_cnt)


def load_commonsense():
    with open('../../datasets/commonsense/train.jsonl', 'r', encoding='utf-8') as f_in:
        f_out = open('../../datasets/commonsense/train_unified.jsonl', 'w', encoding='utf-8')
        for line in f_in.readlines():
            json_obj = json.loads(line)
            context = ''
            question = json_obj['question']['stem']
            answer = json_obj['answerKey']
            for choice in json_obj['question']['choices']:
                if choice['label'] == answer:
                    answer_text = choice['text']
            context += '.'
            answer = json_obj['answerKey']
            json_obj_out = json.dumps({'context': context, 'question': question, 'answer': answer_text}, ensure_ascii=False)
            f_out.write(json_obj_out + '\n')
    with open('../../datasets/commonsense/dev.jsonl', 'r', encoding='utf-8') as f_in:
        dev_f_out = open('../../datasets/commonsense/dev_unified.jsonl', 'w', encoding='utf-8')
        test_f_out = open('../../datasets/commonsense/test_unified.jsonl', 'w', encoding='utf-8')
        dev_cnt = 0
        test_cnt = 0
        for line in f_in.readlines():
            ran_score = random.randint(1, 100)
            json_obj = json.loads(line)
            context = ''
            question = json_obj['question']['stem']
            answer = json_obj['answerKey']
            for choice in json_obj['question']['choices']:
                if choice['label'] == answer:
                    answer_text = choice['text']
            context += '.'
            answer = json_obj['answerKey']
            json_obj_out = json.dumps({'context': context, 'question': question, 'answer': answer_text}, ensure_ascii=False)
            if ran_score > 30:
                dev_f_out.write(json_obj_out + '\n')
                dev_cnt += 1
            else:
                test_f_out.write(json_obj_out + '\n')
                test_cnt += 1
    print(dev_cnt)
    print(test_cnt)


def load_drop():
    with open('../../datasets/drop/train.json', 'r', encoding='utf-8') as f_in:
        f_out = open('../../datasets/drop/train_unified.jsonl', 'w', encoding='utf-8')
        json_obj = json.load(f_in)
        for name, body in json_obj.items():
            context = body['passage']
            for qa in body['qa_pairs']:
                question = qa['question']
                n_answer = qa['answer']['number']
                d_answer = qa['answer']['date']
                s_answer = ''
                for span in qa['answer']['spans']:
                    s_answer += span + ', '
                s_answer = s_answer[:-1]
                if s_answer is not None and len(s_answer) > 0:
                    f_out.write(json.dumps({'context': context, 'question': question, 'answer': s_answer},
                                           ensure_ascii=False) + '\n')
                elif n_answer is not None and len(n_answer) > 0:
                    f_out.write(json.dumps({'context': context, 'question': question, 'answer': n_answer},
                                           ensure_ascii=False) + '\n')

    with open('../../datasets/drop/dev.json', 'r', encoding='utf-8') as f_in:
        dev_out = open('../../datasets/drop/dev_unified.jsonl', 'w', encoding='utf-8')
        test_out = open('../../datasets/drop/test_unified.jsonl', 'w', encoding='utf-8')
        ran_score = random.randint(1, 100)
        json_obj = json.load(f_in)
        dev_cnt = 0
        test_cnt = 0
        for name, body in json_obj.items():
            context = body['passage']
            for qa in body['qa_pairs']:
                ran_score = random.randint(1, 100)
                question = qa['question']
                n_answer = qa['answer']['number']
                d_answer = qa['answer']['date']
                s_answer = ''
                for span in qa['answer']['spans']:
                    s_answer += span + ', '
                s_answer = s_answer[:-1]
                if s_answer is not None and len(s_answer) > 0:
                    if ran_score > 30:
                        dev_out.write(json.dumps({'context': context, 'question': question, 'answer': s_answer},
                                                 ensure_ascii=False) + '\n')
                        dev_cnt += 1
                    else:
                        test_out.write(json.dumps({'context': context, 'question': question, 'answer': s_answer},
                                                  ensure_ascii=False) + '\n')
                        test_cnt += 1
                elif n_answer is not None and len(n_answer) > 0:
                    if ran_score > 30:
                        dev_out.write(json.dumps({'context': context, 'question': question, 'answer': n_answer},
                                                 ensure_ascii=False) + '\n')
                        dev_cnt += 1

                    else:
                        test_out.write(json.dumps({'context': context, 'question': question, 'answer': n_answer},
                                                  ensure_ascii=False) + '\n')
                        test_cnt += 1
    print(dev_cnt)
    print(test_cnt)


def load_MCTest(mode):
    mode = str(mode)
    ans_line_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    df_train = pd.read_csv('../../datasets/MCTest_' + mode + '/mc' + mode + '.train.tsv', sep='\t', header=None)
    df_ans = pd.read_csv('../../datasets/MCTest_' + mode + '/mc' + mode + '.train.ans', sep='\t', header=None)
    f_out = open('../../datasets/MCTest_' + mode + '/train_unified.jsonl', 'w', encoding='utf-8')
    for (i, train_line), (_, ans_line) in zip(df_train.iterrows(), df_ans.iterrows()):
        context = train_line[2]
        context = re.subn(r'\\newline', ' ', context)[0]
        for q_c, a_c in [(3, 0), (8, 1), (13, 2), (18, 3)]:
            answer = ans_line[a_c]
            question = train_line[q_c]
            question = question.split(':')[1]
            question = question[1:]
            ans_text = train_line[q_c + ans_line_dict[answer]]
            f_out.write(json.dumps({'context': context, 'question': question, 'answer': ans_text}, ensure_ascii=False)
                        + '\n')
    f_out.close()
    df_dev = pd.read_csv('../../datasets/MCTest_' + mode + '/mc' + mode + '.dev.tsv', sep='\t', header=None)
    df_ans = pd.read_csv('../../datasets/MCTest_' + mode + '/mc' + mode + '.dev.ans', sep='\t', header=None)
    f_out = open('../../datasets/MCTest_' + mode + '/dev_unified.jsonl', 'w', encoding='utf-8')
    f_out_test = open('../../datasets/MCTest_' + mode + '/test_unified.jsonl', 'w', encoding='utf-8')

    dev_cnt = 0
    test_cnt = 0
    for (i, train_line), (_, ans_line) in zip(df_dev.iterrows(), df_ans.iterrows()):
        context = train_line[2]
        context = re.subn(r'\\newline', ' ', context)[0]
        for q_c, a_c in [(3, 0), (8, 1), (13, 2), (18, 3)]:
            ran_score = random.randint(1, 100)
            answer = ans_line[a_c]
            question = train_line[q_c]
            question = question.split(':')[1]
            question = question[1:]
            ans_text = train_line[q_c + ans_line_dict[answer]]
            if ran_score > 30:
                f_out.write(
                    json.dumps({'context': context, 'question': question, 'answer': ans_text}, ensure_ascii=False)
                    + '\n')
                dev_cnt += 1
            else:
                f_out_test.write(
                    json.dumps({'context': context, 'question': question, 'answer': ans_text}, ensure_ascii=False)
                    + '\n')
                test_cnt += 1
    print(dev_cnt)
    print(test_cnt)


def load_squad():
    json_obj_in = json.load(open('../../datasets/squad1.1/train.json', 'r', encoding='utf-8'))
    f_out = open('../../datasets/squad1.1/train_unified.jsonl', 'w', encoding='utf-8')
    for now_js in json_obj_in['data']:
        for p in now_js['paragraphs']:
            context = p['context']
            for qa in p['qas']:
                question = qa['question']
                for aa in qa['answers']:
                    answer = aa['text']
                    f_out.write(json.dumps({'context': context, 'question': question, 'answer': answer}
                                           , ensure_ascii=False) + '\n')
    f_out.close()
    json_obj_in = json.load(open('../../datasets/squad1.1/dev.json', 'r', encoding='utf-8'))
    f_out_dev = open('../../datasets/squad1.1/dev_unified.jsonl', 'w', encoding='utf-8')
    f_out_test = open('../../datasets/squad1.1/test_unified.jsonl', 'w', encoding='utf-8')
    dev_cnt = 0
    test_cnt = 0
    for now_js in json_obj_in['data']:
        for p in now_js['paragraphs']:
            context = p['context']
            for qa in p['qas']:
                question = qa['question']
                for aa in qa['answers']:
                    ran_score = random.randint(1, 100)
                    answer = aa['text']
                    if ran_score > 30:
                        f_out_dev.write(json.dumps({'context': context, 'question': question, 'answer': answer}
                                                   , ensure_ascii=False) + '\n')
                        dev_cnt += 1
                    else:
                        f_out_test.write(json.dumps({'context': context, 'question': question, 'answer': answer}
                                                    , ensure_ascii=False) + '\n')
                        test_cnt += 1
    print(dev_cnt)
    print(test_cnt)


def load_wikiqa():
    train_df = pd.read_csv('../../datasets/WikiQA/WikiQACorpus/WikiQA-train.tsv', sep='\t')
    f_out = open('../datasets/WikiQA/train_unified.jsonl', 'w', encoding='utf-8')
    for i, train_line in train_df.iterrows():
        if int(train_line['Label']) == 1:
            context = train_line['Sentence']
            question = train_line['Question']


load_boolq()
