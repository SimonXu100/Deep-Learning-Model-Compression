from train_BERT import *

import torch
import torch.nn.utils.prune as prune

import argparse

parser = argparse.ArgumentParser(
    description='PyTorch BERT Prune & Retraining')
parser.add_argument('--prune_fraction', type=float, default=0.5,
                    help='Fraction of parameters to prune each iteration')
parser.add_argument('--iterations', type=int, default=6,
                    help='Number of iterations for iterative pruning')
parser.add_argument('--random_prune', action='store_true',
                    help='Randomly select connections to prune')


def finetune(model, tokenizer, num_epochs, trainloader, testloader, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    test_accus = train(model, tokenizer, device, trainloader, testloader,
                       optimizer, num_epochs, None, scheduler=None)
    train_accu = evaluate(model, tokenizer, trainloader, device)
    print("Finetune test accus:", test_accus, file=output_file)
    return test_accus[-1], train_accu


def get_mask_sum(module):
    mask_sum = 0
    children = list(module.named_children())
    if len(children) > 0:
        for name, module in children:
            mask_sum += get_mask_sum(module)
    else:
        for name, mask in module.named_buffers():
            mask_sum += mask.sum().item()
    return mask_sum


# Recursively find the module with attrs2prune(weight and bias) and collect those modules
def get_parameters_to_prune(module, attrs2prune, module_name="model"):
    parameters_to_prune = []
    parameters_names = []
    children = list(module.named_children())
    if len(children) > 0:
        for name, module in children:
            output = get_parameters_to_prune(module, attrs2prune, module_name=module_name+"."+name)
            parameters_to_prune += output[0]
            parameters_names += output[1]
    else:
        for name, param in module.named_parameters():
            if name in attrs2prune:
                parameters_to_prune.append((module, name))
                parameters_names.append(module_name+"."+name)
    return parameters_to_prune, parameters_names


if __name__ == "__main__":
    MODEL_NAME = 'bert-base-uncased'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If you prefer print outputs to console, change output_file to None
    output_file = open("output.txt", "w")

    args = parser.parse_args()
    print(args, file=output_file)

    batch_size = 16
    chkpt_dir = "./BERT_final_chkpt/"
    prune_iteration = args.iterations
    prune_fraction = args.prune_fraction

    np.random.seed(0)
    torch.manual_seed(0)

    if args.random_prune:
        pruning_method = prune.RandomUnstructured
    else:
        pruning_method = prune.L1Unstructured

    # Load dataset and create dataloader
    dataset = load_dataset("glue", "sst2")
    trainset, testset = dataset["train"], dataset["validation"]
    print(dataset, file=output_file)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # Initialize the tokenizer for preprocessing input of BERT
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # Load pretrained bert model from Transformers package
    model = BertForSequenceClassification.from_pretrained(chkpt_dir)
    model.to(DEVICE)
    print(model, file=output_file)

    test_accu, train_accu = evaluate(model, tokenizer, testloader, DEVICE), \
                            evaluate(model, tokenizer, trainloader, DEVICE)
    print(test_accu, train_accu, file=output_file)

    frac_list = [100]
    test_accus_prune = [test_accu]
    train_accus_prune = [train_accu]
    test_accus_prune_finetuned = [test_accu]
    train_accus_prune_finetuned = [train_accu]
    parameters_to_prune, parameters_names = get_parameters_to_prune(model, ("weight", "bias"))
    print(pruning_method, file=output_file)
    for i in range(prune_iteration):
        if output_file != None:
            print("=========================Iteration %i ==========================" % (i + 1))
        print("=========================Iteration %i =========================="%(i+1), file=output_file)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=prune_fraction,
        )
        frac_list.append(frac_list[-1]*(1-prune_fraction))

        test_accu, train_accu = evaluate(model, tokenizer, testloader, DEVICE), \
                                evaluate(model, tokenizer,trainloader, DEVICE)
        print("Performance before finetuning:", file=output_file)
        print("Test accuracy:", test_accu, file=output_file)
        print("Training accuracy:", train_accu, file=output_file)
        test_accus_prune.append(test_accu)
        train_accus_prune.append(train_accu)

        test_accu, train_accu = finetune(model, tokenizer, 1, trainloader, testloader, DEVICE)
        print("Performance after finetuning:", file=output_file)
        print("Test accuracy:", test_accu, file=output_file)
        print("Training accuracy:", train_accu, file=output_file)
        test_accus_prune_finetuned.append(test_accu)
        train_accus_prune_finetuned.append(train_accu)

        sum_zero_weight = 0
        sum_weight = 0
        for i in range(len(parameters_to_prune)):
            module, parameter = parameters_to_prune[i]
            cur_zero_weight = float(torch.sum(getattr(module,parameter) == 0))
            cur_weight = float(getattr(module,parameter).nelement())
            print("Sparsity in {}: {:.2f}%".format(parameters_names[i],
                100. * cur_zero_weight/cur_weight), file=output_file)
            sum_zero_weight+=cur_zero_weight
            sum_weight += cur_weight
        print("Global sparsity: {:.2f}%".format(100. * sum_zero_weight/sum_weight), file=output_file)
        print(sum_zero_weight, get_mask_sum(model), sum_weight, file=output_file)

    output_file.close()
    result = np.vstack((frac_list, test_accus_prune, train_accus_prune, test_accus_prune_finetuned, train_accus_prune_finetuned))
    np.savetxt("BERT_unstructured_performance.txt", result)

