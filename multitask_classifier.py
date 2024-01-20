import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask

from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss

TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    print(f"Seed number is {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO_DONE

        self.attention_layer = AttentionLayer(config.hidden_size)

        # Sentiment task
        self.classification_layer  = torch.nn.Linear(config.hidden_size, config.hidden_size)    # create linear layer
        self.classification_layer1 = torch.nn.Linear(config.hidden_size, config.hidden_size)    # create linear layer
        self.classification_layer2 = torch.nn.Linear(config.hidden_size, config.hidden_size)    # create linear layer
        self.classification_layer_out = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)      # create linear layer
        
        self.classification_batch_norm1 = nn.BatchNorm1d()          # create batch normalization
        self.classification_batch_norm2 = nn.BatchNorm1d()          # create batch normalization
        self.classification_batch_norm3 = nn.BatchNorm1d()          # create batch normalization

        # Paraphrase task
        self.paraphrase_layer  = torch.nn.Linear(config.hidden_size, config.hidden_size)  # create linear layer
        self.paraphrase_layer1 = torch.nn.Linear(config.hidden_size, config.hidden_size)  # create linear layer
        self.paraphrase_layer2 = torch.nn.Linear(config.hidden_size, config.hidden_size)  # create linear layer
        self.paraphrase_layer3 = torch.nn.Linear(config.hidden_size, config.hidden_size)  # create linear layer
        self.paraphrase_layer_out = nn.Linear(config.hidden_size, 2)                      # create linear layer

        self.paraphrase_batch_norm1 = nn.BatchNorm1d()          # create batch normalization
        self.paraphrase_batch_norm2 = nn.BatchNorm1d()          # create batch normalization
        self.paraphrase_batch_norm3 = nn.BatchNorm1d()          # create batch normalization


        #self.classification_dropout = torch.nn.Dropout(config.hidden_dropout_prob) # create dropout
        self.paraphrase_dropout = torch.nn.Dropout(config.hidden_dropout_prob)      # create dropout
        self.similarity_dropout = torch.nn.Dropout(config.hidden_dropout_prob)      # create dropout
    
        #raise NotImplementedError


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO_DONE

        pooled_output = self.bert(input_ids,attention_mask)['pooler_output']            # get minBERT embeddings
        pooled_output = self.attention_layer(pooled_output)                             # use attention layer
        return pooled_output
        #raise NotImplementedError



    def predict_sentiment(self, input_ids, attention_mask,is_eval=False):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO_DONE
        pooled_output = self.forward(input_ids,attention_mask)                                              # get minBERT embedings
        pooled_output = F.relu(self.classification_batch_norm1(self.classification_layer(pooled_output)))   # use linear layer, batch normalization, ReLU function
        pooled_output = F.relu(self.classification_batch_norm2(self.classification_layer1(pooled_output)))  # use linear layer, batch normalization, ReLU function
        pooled_output = F.relu(self.classification_batch_norm3(self.classification_layer2(pooled_output)))  # use linear layer, batch normalization, ReLU function
        
        logits = self.classification_layer_out(pooled_output)   # use linear layer
        return logits
        #raise NotImplementedError


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO_DONE
        pooled_output_1 = self.forward(input_ids_1,attention_mask_1)    # get minBERT embeddings
        pooled_output_2 = self.forward(input_ids_2,attention_mask_2)    # get minBERT embeddings

        pooled_output_1 = self.paraphrase_layer(pooled_output_1)        # use linear layer
        pooled_output_2 = self.paraphrase_layer(pooled_output_2)        # use linear layer
        
        #NOTE: Please see 'multitask_classifier_with_all_tasks.py' for more cosine similarity and Siamese network 
        # cat 
        #cat = torch.cat([pooled_output_1,pooled_output_2,torch.abs(pooled_output_1,pooled_output_2)],dim=1)
        # cosine similarity
        #logits = F.cosine_similarity(pooled_output_1, pooled_output_2, dim=-1)
        #pooled_output = self.paraphrase_dropout(cat)
        #logits = self.paraphrase_layer(pooled_output)
        
        pooled_output = F.relu(self.paraphrase_batch_norm1(self.paraphrase_layer1(pooled_output)))      # use linear layer, batch normalization, ReLU function 
        pooled_output = F.relu(self.paraphrase_batch_norm2(self.paraphrase_layer2(pooled_output)))      # use linear layer, batch normalization, ReLU function
        pooled_output = F.relu(self.paraphrase_batch_norm3(self.paraphrase_layer3(pooled_output)))      # use linear layer, batch normalization, ReLU function

        logits = self.paraphrase_layer_out(pooled_output)                                               # use linear layer
        return logits
        #raise NotImplementedError


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO_DONE

        pooled_output_1 = self.forward(input_ids_1,attention_mask_1) # get minBERT embeddings
        pooled_output_2 = self.forward(input_ids_2,attention_mask_2) # get minBERT embeddings


        #NOTE: Please see 'multitask_classifier_with_all_tasks.py' for more cosine similarity and Siamese network 
        # cat
        #cat = torch.cat([pooled_output_1,pooled_output_2],dim=1)
        # cosine similarity
        logits = F.cosine_similarity(pooled_output_1, pooled_output_2)              # use cosine similarity
        #logits = F.cosine_similarity(pooled_output_1, pooled_output_2, dim=-1)
        #pooled_output = self.similarity_dropout(cat)
        #logits = self.similarity_layer(pooled_output)

        return logits
        #raise NotImplementedError



def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


#CHANGED**** @ 19.12.2023
# NOTE: the below funtions were borrowed from Github
def symmetrized_kl_divergence(p, q):

    return F.kl_div(p, q, reduction='batchmean') + F.kl_div(q, p, reduction='batchmean')

def smoothness_regularizer(predictions, targets, epsilon):
    pairwise_distances = torch.norm(predictions - predictions.unsqueeze(1), dim=-1)
    mask = pairwise_distances <= epsilon
    max_distances = torch.max(mask * pairwise_distances, dim=1).values
    return torch.sum(max_distances)
#CHANGED****

## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)
        
            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
        
            
            #CHANGED**** @ 19.12.2023
            # NOTE: The below lines is an attempt for SMART loss, but this is not valid implementation
            #loss += smart_loss
            
            #predictions = F.softmax(logits,dim=-1)
            #targets = b_labels.view(-1)

            # Smoothness-inducing adversarial regularizer
            #epsilon=10**(-5)
            #lambda_s=1
            #mu=1
            #bregman_div = symmetrized_kl_divergence(predictions, predictions.detach())
            #smoothness_reg = smoothness_regularizer(predictions, targets, epsilon)
            # Total loss for VBPP method
            #loss = loss + mu * bregman_div + lambda_s * smoothness_regularizer(predictions, targets, epsilon)
            # Compute gradients
            #gradients = grad(total_loss, model.parameters(), create_graph=True)
        
            # Update model parameters using gradient descent
            #with torch.no_grad():
            #    for param, gradient in zip(model.parameters(), gradients):
            #        param -= mu * gradient
            #CHANGED****
    
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
