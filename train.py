



def train(dataloader, model, tokenizer):


    for batch in dataloader:
        context = batch['context']
        question = batch['question']
        answer = batch['answer']
