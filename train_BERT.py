import numpy as np

from datasets import load_dataset
import transformers
from transformers import BertTokenizer, BertForSequenceClassification

import torch


def train(model, tokenizer, device, train_loader, test_loader,
          optimizer, max_epoch, chkpt_dir, scheduler = None):
    test_accus = []
    for epoch in range(max_epoch):
        model.train()
        for i, data in enumerate(train_loader):
            # Prepare the batch for BERT input
            tokenized_x = tokenizer(data["sentence"], padding=True, truncation=True, return_tensors="pt")
            input_ids = tokenized_x["input_ids"].to(device)
            token_type_ids = tokenized_x["token_type_ids"].to(device)
            attention_mask = tokenized_x["attention_mask"].to(device)
            labels = data['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                _, predicted = torch.max(outputs.logits, 1)
                accu = (predicted == labels).double().sum().item() / labels.shape[
                    0] * 100
                print('\r', "Epoch:", epoch + 1, "Iter:", i + 1, "loss =",
                      loss.data.item(), "accu =", "{:.2f}%".format(accu),
                      end='')

        if scheduler:
            scheduler.step()

        print('')
        cur_test_accu = evaluate(model, tokenizer, test_loader, device)
        print("Test accu = {:.2f}%".format(cur_test_accu * 100))
        if chkpt_dir and (len(test_accus) == 0 or cur_test_accu > np.max(test_accus)):
            print("Checkpoint saved at epoch", epoch)
            model.save_pretrained(chkpt_dir)
        test_accus.append(cur_test_accu)

    return test_accus


def evaluate(model, tokenizer, test_loader, device):
    model.eval()
    with torch.no_grad():
        # TODO
        num_correct = 0
        for i, data in enumerate(test_loader):
            tokenized_x = tokenizer(data["sentence"], padding=True,
                                    truncation=True, return_tensors="pt")
            input_ids = tokenized_x["input_ids"].to(device)
            token_type_ids = tokenized_x["token_type_ids"].to(device)
            attention_mask = tokenized_x["attention_mask"].to(device)
            labels = data['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)
            _, predicted = torch.max(outputs.logits, 1)
            cur_correct = (predicted == labels).double().sum().item()
            num_correct += cur_correct
            if i % 10 == 0:
                accu = cur_correct / labels.shape[0] * 100
                print('\r', "Iter:", i + 1, "accu =", "{:.2f}%".format(accu),end='')
        print('')
        test_accu = num_correct / len(test_loader.dataset)
        return test_accu


if __name__ == "__main__":
    MODEL_NAME = 'bert-base-uncased'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    chkpt_dir = "BERT_chkpt/"
    max_epoch = 10

    dataset = load_dataset("glue", "sst2")
    print(dataset)
    trainset, testset = dataset["train"], dataset["validation"]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)
    print(model)

    optimizer = transformers.optimization.AdamW(model.parameters(), lr=1e-5)
    test_accus = train(model, tokenizer, DEVICE, trainloader, testloader,
                       optimizer, max_epoch, chkpt_dir)
    print(test_accus)


    model_loaded = BertForSequenceClassification.from_pretrained(chkpt_dir)
    model_loaded.to(DEVICE)
    print(model_loaded)
    print(evaluate(model_loaded, tokenizer, testloader, DEVICE))
