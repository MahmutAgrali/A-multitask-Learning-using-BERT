import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask
from pcgrad import PCGrad

from sklearn.utils import class_weight 
import copy

#CHANGED @16.01.2024
from triplet_loss import TripletLoss    # import triplet loss implementation
#CHANGED

TQDM_DISABLE=False

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

#CHANGED**** @ 05.01.2023
class TripletDataset(Dataset):
    """
    @ Dataset class for Triplet loss 
    """
    def __init__(self, data_ids,data_masks,data_labels,device):
        self.data_ids = data_ids        # save ids 
        self.data_masks = data_masks    # save masks
        self.data_labels = data_labels  # save labels   
        self.device = device            # save device param
    
    def __len__(self):
        return len(self.data_ids)    # return dataset size

    def __getitem__(self, idx,labels):
        self.data_ids             
        positive_ids = list()       # create lists for ids
        positive_masks = list()     # create lists for masks

        negative_ids = list()       # create lists for ids
        negative_masks = list()     # create lists for masks
        
        for label in labels:                                                                # track all labels
            positive_label_idx = -1                                                         # initilize positive_label_idx
            label = label.item()                                                            # get tensor value
            while positive_label_idx != label:                                              # run when positive_label_idx is not label
                positive_label_idx = torch.randint(0, len(self.data_labels), (1,)).item()   # randomly select an integer for positives
            
            positive_id = self.data_ids[positive_label_idx]            # get positive id using the random index
            positive_mask = self.data_masks[positive_label_idx]        # get positive mask using the random index
            positive_ids.append(positive_id)                           # add to list id
            positive_masks.append(positive_mask)                       # add to list mask

            negative_label_idx = label                                                      # initilize negative_label_idx
            while negative_label_idx == label:                                              # run when negative_label_idx is label
                negative_label_idx = torch.randint(0, len(self.data_labels), (1,)).item()   # get random integer for negative label index

            negative_id = self.data_ids[negative_label_idx]         # get negative id using the random index
            negative_mask = self.data_masks[negative_label_idx]     # get negative mask using the random index 
            negative_ids.append(negative_id)                        # add to list id
            negative_masks.append(negative_mask)                    # add to list mask

        positive_ids = torch.stack(positive_ids).to(self.device)         # decrease dimentionality of positive ids
        positive_masks = torch.stack(positive_masks).to(self.device)     # decrease dimentionality of positive mask
        negative_ids = torch.stack(negative_ids).to(self.device)         # decrease dimentionality of negative ids
        negative_masks = torch.stack(negative_masks).to(self.device)     # decrease dimentionality of negative mask

        return positive_ids, positive_masks, negative_ids, negative_masks 
#CHANGED****

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
        if not args.use_bert_large:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.bert = BertModel.from_pretrained('bert-large-uncased')


        self.config = config
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True


        ### TODO_DONE
        #CHANGED @ 12.01.2023
        
        if self.config.use_bert_large: # check if largeBERT is used 
            self.BERT_HIDDEN_SIZE_ = 1024   # get hidden size for largeBERT

            self.conv1d_1 = nn.Conv1d(in_channels=self.BERT_HIDDEN_SIZE_, out_channels=(self.BERT_HIDDEN_SIZE_)//4, kernel_size=2,padding='same')       # create convolutional layer for largeBERT
            self.conv1d_2 = nn.Conv1d(in_channels=self.BERT_HIDDEN_SIZE_, out_channels=(self.BERT_HIDDEN_SIZE_)//4, kernel_size=3,padding='same')       # create convolutional layer for largeBERT
            self.conv1d_3 = nn.Conv1d(in_channels=self.BERT_HIDDEN_SIZE_, out_channels=(self.BERT_HIDDEN_SIZE_)//4, kernel_size=4,padding='same')       # create convolutional layer for largeBERT
            self.conv1d_4 = nn.Conv1d(in_channels=self.BERT_HIDDEN_SIZE_, out_channels=(self.BERT_HIDDEN_SIZE_)//4, kernel_size=5,padding='same')       # create convolutional layer for largeBERT

        else:
            self.BERT_HIDDEN_SIZE_ = BERT_HIDDEN_SIZE # get hiddin size for smallBERT, 768
            
            self.conv1d_1 = nn.Conv1d(in_channels=self.BERT_HIDDEN_SIZE_, out_channels=(self.BERT_HIDDEN_SIZE_)//3, kernel_size=2,padding='same')       # create convolutional layer for smallBERT
            self.conv1d_2 = nn.Conv1d(in_channels=self.BERT_HIDDEN_SIZE_, out_channels=(self.BERT_HIDDEN_SIZE_)//3, kernel_size=3,padding='same')       # create convolutional layer for smallBERT
            self.conv1d_3 = nn.Conv1d(in_channels=self.BERT_HIDDEN_SIZE_, out_channels=(self.BERT_HIDDEN_SIZE_)//3, kernel_size=4,padding='same')       # create convolutional layer for smallBERT        

        
        if self.config.use_bert_large: # check if largeBERT is used
            self.attention_layer = nn.MultiheadAttention(embed_dim=1024, num_heads=1)               # create multi head attention for largeBERT
        else:
            self.attention_layer = nn.MultiheadAttention(embed_dim=BERT_HIDDEN_SIZE, num_heads=1)   # create multi head attention for smallBERT

        # Sentiment task
        if self.config.use_bert_large: # check if largeBERT is used
            self.classification_layer  = nn.Linear(1024, BERT_HIDDEN_SIZE)                  # create linear layer for largeBERT
        else:
            self.classification_layer  = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)      # create linear layer for smallBERT

        self.classification_layer1 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)          # create linear layer
        self.classification_layer2 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)          # create linear layer
        if self.config.use_bert_large:
            self.classification_layer3 = nn.Linear(1024, BERT_HIDDEN_SIZE)                  # create linear layer for largeBERT
        else:
            self.classification_layer3 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)      # create linear layer for smallBERT
        self.classification_layer_out = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)    # create linear layer
        
        self.classification_batch_norm1 = nn.BatchNorm1d(BERT_HIDDEN_SIZE)                  # create batch normalization
        self.classification_batch_norm2 = nn.BatchNorm1d(BERT_HIDDEN_SIZE)                  # create batch normalization
        self.classification_batch_norm3 = nn.BatchNorm1d(BERT_HIDDEN_SIZE)                  # create batch normalization
        self.classification_batch_norm4 = nn.BatchNorm1d(BERT_HIDDEN_SIZE)                  # create batch normalization
        self.classification_dropout = nn.Dropout(p=config.hidden_dropout_prob)              # create dropout

        # Paraphrase task
        if self.config.use_bert_large: # check if largeBERT is used
            self.paraphrase_layer  = nn.Linear(1024, BERT_HIDDEN_SIZE)                  # create linear layer 
        else:
            self.paraphrase_layer  = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)      # create linear layer 
        self.paraphrase_layer1 = nn.Linear(BERT_HIDDEN_SIZE*3, BERT_HIDDEN_SIZE)        # create linear layer 
        self.paraphrase_layer2 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)          # create linear layer 
        self.paraphrase_layer3 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)          # create linear layer 
        
    
        if self.config.use_cosine_similarity: # check if cosine similarity is used
            self.paraphrase_layer_out = nn.Linear(BERT_HIDDEN_SIZE, 1)          # create linear layer for cosine similarity
        else:
            if self.config.use_bert_large: # check if largeBERT is used
                self.paraphrase_layer_out = nn.Linear(1024*2, 1)                # create linear layer  for largeBERT
                
                self.similarity_layer = nn.Linear(1024*2, BERT_HIDDEN_SIZE)     # create linear layer for largeBERT 
                self.similarity_batch_norm = nn.BatchNorm1d(BERT_HIDDEN_SIZE)   # create batch normalization for largeBERT
                self.similarity_layer_out = nn.Linear(BERT_HIDDEN_SIZE, 1)      # create classification layer for largeBERT

            else:
                self.paraphrase_layer_out = nn.Linear(BERT_HIDDEN_SIZE*2, 1)    # create linear layer for smallBERT
                self.similarity_layer_out = nn.Linear(BERT_HIDDEN_SIZE*2, 1)    # create linear layer for smallBERT

        self.paraphrase_batch_norm1 = nn.BatchNorm1d(BERT_HIDDEN_SIZE)          # create batch normalization for largeBERT
        self.paraphrase_batch_norm2 = nn.BatchNorm1d(BERT_HIDDEN_SIZE)          # create batch normalization for largeBERT
        self.paraphrase_batch_norm3 = nn.BatchNorm1d(BERT_HIDDEN_SIZE)          # create batch normalization for largeBERT
        #CHANGED     
        #raise NotImplementedError


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO_DONE

        pooled_output = self.bert(input_ids,attention_mask)['pooler_output']                        # get minBERT embeddings
        if self.config.use_attention_layer:                                                         # check if attention layer is used
            pooled_output = self.attention_layer(pooled_output,pooled_output,pooled_output)[0]      # use attention layer afteer the minBERT embeddings
        
        if self.config.use_conv:                                                                    # check if convolutional layers are used
            pooled_output = pooled_output.unsqueeze(2)                                              # decrease dimention of the minBERT embeddings

            if self.config.use_bert_large:                                                          # check if largeBERT is used
                conv_out_1 = self.conv1d_1(pooled_output)                                           # used convolutional layer for largeBERT kernel size=2
                conv_out_2 = self.conv1d_2(pooled_output)                                           # used convolutional layer for largeBERT kernel size=3
                conv_out_3 = self.conv1d_3(pooled_output)                                           # used convolutional layer for largeBERT kernel size=4
                conv_out_4 = self.conv1d_4(pooled_output)                                           # used convolutional layer for largeBERT kernel size=5
                pooled_output = torch.cat([conv_out_1, conv_out_2,conv_out_3,conv_out_4], dim=1)    # concatenete the output embeeddings of the convolutional layers

            else:
                conv_out_1 = self.conv1d_1(pooled_output)                                           # used convolutional layer for smallBERT kernel size=2
                conv_out_2 = self.conv1d_2(pooled_output)                                           # used convolutional layer for smallBERT kernel size=3
                conv_out_3 = self.conv1d_3(pooled_output)                                           # used convolutional layer for smallBERT kernel size=4
                pooled_output = torch.cat([conv_out_1, conv_out_2,conv_out_3], dim=1)               # concatenete the output embeeddings of the convolutional layers

            pooled_output = pooled_output.squeeze(2)                                                # increase dimention of the embeddings
            

        return pooled_output
        #raise NotImplementedError

    #CHANGED**** @ 05.01.2023
    def triplet_loss_fn(self,anchor, positive, negative, margin = 1.0, is_mean=True):
        def calc_euclidean(x1, x2):
            return (x1 - x2).pow(2).sum(1)              # calculate euclidean distance
        anchor   = F.normalize(anchor, p=2, dim=1)      # normalize anchor embeddings
        positive = F.normalize(positive, p=2, dim=1)    # normalize positive embeddings
        negative = F.normalize(negative, p=2, dim=1)    # normalize negative embeddings

        distance_positive = calc_euclidean(anchor, positive)                        # calculate euclidean distance for anchor-positive pair 
        distance_negative = calc_euclidean(anchor, negative)                        # calculate euclidean distance for anchor-negative pair
        losses = torch.clamp(distance_positive-distance_negative+margin,min=0.0)    # return loss value as 0.0 if 'istance_positive-distance_negative+margin' is lower than zero
        return losses.mean()                                                        # return mean loss value, also 'if is_mean else losses.sum()' can be used
        
    def triplet_loss_sentiment_v0(self,input_ids,attention_masks,labels,triplet_dataset):

        positive_ids, positive_masks, negative_ids, negative_masks = triplet_dataset.__getitem__(input_ids,labels)    # get ids and masks for negatives and positives form the dataset

        input_ids_pooled_output = self.forward(input_ids, attention_masks)                                            # get the embedding for anchor 
        positive_pooled_output = self.forward(positive_ids, positive_masks)                                           # get the embedding for positives
        negative_pooled_output = self.forward(negative_ids, negative_masks)                                           # get the embedding for negatives

        triplet_loss = self.triplet_loss_fn(input_ids_pooled_output,positive_pooled_output, negative_pooled_output)  # caluclatte triplet loss

        return triplet_loss
    
    def triplet_loss_sentiment_v00(self,input_ids,attention_masks,labels,triplet_dataset):
        
        
        batch_embeddings = self.forward(input_ids, attention_masks)                 # get the embeddings for anchor, positives, and negatives                                                   
        
        hard_positive_indices = torch.argmin(F.pairwise_distance(batch_embeddings, batch_embeddings[torch.randperm(len(batch_embeddings))],keepdim=True), dim=0)    # chose hard positive indices by calculating pairwise distance
        
        hard_positive_embeddings = batch_embeddings[hard_positive_indices]                                                      # get hard positive embeddings via hard positive indices 
    
        distances = F.pairwise_distance(batch_embeddings, batch_embeddings[torch.randperm(len(batch_embeddings))],keepdim=True) # get pairwise distance for negative indices
        hard_negative_indices = torch.argmax(distances, dim=0)                                                                  # get negative indices

        hard_negative_embeddings = batch_embeddings[hard_negative_indices]                                                      # get hard negative embeddings
        
        return self.triplet_loss_fn(batch_embeddings, hard_positive_embeddings, hard_negative_embeddings, margin=0.5)           # calculate triplet loss
        #return F.margin_ranking_loss(batch_embeddings, hard_positive_embeddings, hard_negative_embeddings, margin=0.5)         # calculate margin ranking loss
    #CHANGED****

    #CHANGED**** @ 12.01.2024
    def __setssttraindata__(self,sst_train_data):
        self.sst_train_data = sst_train_data                #   set sst_train_data into the class as an class object

    def __setdatasetloader__(self,triplet_dataloader):
        self.triplet_dataloader = triplet_dataloader         #   set sst_train_data into the class as an class object

    def __getdataset__(self):
        try:
            batch = next(self.triplet_dataloader)                       # get next batch
            return batch
        except:
            self.triplet_dataloader = DataLoader(self.sst_train_data, shuffle=True, batch_size=self.config.batch_size*4,
                                      collate_fn=self.sst_train_data.collate_fn)        # create new dataloader when there is no remaining batch 
            self.triplet_dataloader = iter(self.triplet_dataloader)                     # convert dataloader as an iterable object

            batch = next(self.triplet_dataloader)                                       # get batch
            return batch
        
    def __selectsamples__(self,idxs,masks,triplet_ids_sst, triplet_mask_sst, triplet_labels_sst,labels):

        hard_positive_embs = torch.zeros(idxs.shape[0],self.BERT_HIDDEN_SIZE_).to(self.config.device)       # create zeros tensor for hard positive embeddings
        hard_negative_embs = torch.zeros(idxs.shape[0],self.BERT_HIDDEN_SIZE_).to(self.config.device)       # create zeros tensor for hard negative embeddings
        
        arch_embeddings = self.forward(idxs, masks)                                                         # get archor embeddings
        embeddings = self.forward(triplet_ids_sst, triplet_mask_sst)                                        # get embeddings for triplet samples
        
        for i, arch_emb in enumerate(arch_embeddings):
            max_similarity = 0                                                              # initilize max_similarity
            min_similarity = 1                                                              # initilize min_similarity
            for  emb in embeddings:
                similarity = F.cosine_similarity(arch_emb.unsqueeze(0),emb.unsqueeze(0))    # calculate cosine similarity to select hard positive and negative  
                if similarity > max_similarity:                                             # check max similarity
                    max_similarity = similarity                                             # exhange max_similarity <->similarity
                    hard_positive_embs[i,:] = emb                                           # add embedding into hard positive embedding tensor

                if similarity < min_similarity:                                             # check min similarity
                    min_similarity = similarity                                             # exhange min_similarity <->similarity
                    hard_negative_embs[i,:] = emb                                           # add embedding into hard negative embedding tensor
        return arch_embeddings, hard_positive_embs, hard_negative_embs                

    def triplet_loss_sentiment(self,input_ids,attention_masks,labels,margin = 1.0,is_mean=True):
        triplet_batch_sst = self.__getdataset__()                                                       # get dataset
        triplet_ids_sst, triplet_mask_sst, triplet_labels_sst = (triplet_batch_sst['token_ids'],triplet_batch_sst['attention_mask'], triplet_batch_sst['labels']) # get ids, masks, and labels from the batch
        triplet_ids_sst = triplet_ids_sst.to(self.config.device)        # move the tensors to GPU or CPU
        triplet_mask_sst = triplet_mask_sst.to(self.config.device)      # move the tensors to GPU or CPU
        triplet_labels_sst = triplet_labels_sst.to(self.config.device)  # move the tensors to GPU or CPU
        
        arch_embeddings, hard_positive_embeddings, hard_negative_embeddings = self.__selectsamples__(input_ids,attention_masks,triplet_ids_sst, triplet_mask_sst, triplet_labels_sst,labels) # select hard positive and hard negative samples

    
        if self.config.use_contrastive_loss:                                                                           # check if contrastive loss is used 
            def constrastive_loss_fn(arch_embeddings, hard_positive_embeddings, hard_negative_embeddings,labels):
                distance_positive = F.pairwise_distance(arch_embeddings, hard_positive_embeddings)                                       # calculate pairwise distance for hard positives
                distance_negative = F.pairwise_distance(arch_embeddings, hard_negative_embeddings)                                       # calculate pairwise distance for hard negatives
                margin = 0.5                                                                                                             # define margin variable
                loss = 0.5 * (labels * distance_positive**2 + (1 - labels) * torch.clamp(margin - distance_negative, min=0)**2).mean()   # calculate contrastive loss
                return loss
            return constrastive_loss_fn(arch_embeddings, hard_positive_embeddings, hard_negative_embeddings,labels)                      # calculate contrastive loss
            
            
        else:
            return self.triplet_loss_fn(arch_embeddings, hard_positive_embeddings, hard_negative_embeddings, margin=margin,is_mean=is_mean) # calculate triplet loss

    #CHANGED****
    
    
    #CHANGED**** @ 01.01.2024
    def compute_contrastive_loss(self, anchor, positives, negatives):
        def exp_sim(x,x_,temperature=0.05):
            return torch.exp(F.cosine_similarity(x,x_)/temperature)                                                                         # calculate eponantial similarity 
        contrastive_loss = -torch.log(exp_sim(anchor,positives)/ ( torch.sum(exp_sim(anchor,positives) + exp_sim(anchor,negatives)) ))      # calculate contrastive loss
        return torch.mean(contrastive_loss)                                                                                                 # retun mean value of contrastive loss

    def contrastive_loss_sentiment(self,input_ids,attention_masks,labels,all_ids,all_masks,all_labels,device):
        rand = torch.randperm(len(input_ids))                                                # get random indices

        input_ids_pooled_output = self.forward(input_ids, attention_masks)                   # compute the embeddings for the positive pair

        similar_input_ids = input_ids[rand]                                                  # get ids for postive embeddings
        similar_attention_masks = attention_masks[rand]                                      # get masks for positive embeddings

        similar_pooled_output = self.forward(similar_input_ids, similar_attention_masks)     # compute the embeddings for the positive pair

        different_input_ids = []                                            # create a list for negative ids
        different_attention_masks = []                                      # create a list for negative masks
        for i,la in enumerate(labels):
            sample_ids = all_ids[all_labels != la]                          # get sample ids that is different from achor label
            sample_masks = all_masks[all_labels != la]                      # get sample masks that is different from achor label
            sample_labels = all_labels[all_labels != la]                    # get sample labels that is different from achor label

            random_index = torch.randint(0, len(sample_ids), size=(1,))     # get random index 
            
            sample_id = sample_ids[random_index][0]                         # get sample id using random index
            sample_mask = sample_masks[random_index][0]                     # get sample mask using random index

            different_input_ids.append(sample_id)                           # add sample id to the list 
            different_attention_masks.append(sample_mask)#[rand])           # add sample mask to the list

        different_input_ids = nn.utils.rnn.pad_sequence(different_input_ids, batch_first=True)                  # create a sequencee for negative ids
        different_attention_masks = nn.utils.rnn.pad_sequence(different_attention_masks, batch_first=True)      # create a sequence for negative masks

        different_input_ids, different_attention_masks = different_input_ids.reshape(8,-1), different_attention_masks.reshape((8,-1))   # reshape ids and masks
        different_pooled_output = self.forward(different_input_ids, different_attention_masks)                                          # get embedings for negative samples

        contrastive_loss = self.compute_contrastive_loss(input_ids_pooled_output,similar_pooled_output, different_pooled_output)        # calculate contrastive loss
        return contrastive_loss
        

    def predict_sentiment(self, input_ids, attention_mask,is_eval=False):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        
        ### TODO_DONE
        pooled_output = self.forward(input_ids,attention_mask) # get minBERT embeddings, shape::(8,1024)

        if self.config.use_linear_layers:                                                                       # check if linear layers are used

            pooled_output = F.relu(self.classification_batch_norm1(self.classification_layer(pooled_output)))   # use linear layer, batch normalization and relu function
            pooled_output = F.relu(self.classification_batch_norm2(self.classification_layer1(pooled_output)))  # use linear layer, batch normalization and relu function

        if self.config.use_bert_large and not self.config.use_linear_layers:                                     # check if largeBERT and linear layers are used

            pooled_output = F.relu(self.classification_batch_norm4(self.classification_layer3(pooled_output)))  # use linear layer, batch normalization and relu function
        logits = self.classification_layer_out(pooled_output)                                                   # use linear layer (classification layer)
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
        pooled_output_1 = self.forward(input_ids_1,attention_mask_1)         # get embeddings for first sentences
        pooled_output_2 = self.forward(input_ids_2,attention_mask_2)         # get embeddings for second sentences
    
        if self.config.use_cosine_similarity:                                # check if cosine similarity is used
            # NOTE: Please consider that is the code in the below is called Siamies network. The reason used with use_cosine_similarity variable is to avoid the use of more parameter 

            pooled_output_1 = self.paraphrase_layer(pooled_output_1)         # use linear layer for first sentences
            pooled_output_2 = self.paraphrase_layer(pooled_output_2)         # use linear layer for second sentences
            
            cat = torch.cat([pooled_output_1,pooled_output_2,torch.abs(pooled_output_1-pooled_output_2)],dim=1)     # concatinate the embeddings for first and second sentences, as well as their substraction 
    
            pooled_output = F.relu(self.paraphrase_batch_norm1(self.paraphrase_layer1(cat)))                # use linear layer, batch normalization and relu function
            pooled_output = F.relu(self.paraphrase_batch_norm2(self.paraphrase_layer2(pooled_output)))      # use linear layer, batch normalization and relu function
        else:
            pooled_output = torch.cat([pooled_output_1,pooled_output_2],dim=1)                              # concatinate the embeddings of two sentencees

        logits = self.paraphrase_layer_out(pooled_output)[:,0]                                              # use linear layer for classification
        
        return logits
        #raise NotImplementedError


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO_DONE

        pooled_output_1 = self.forward(input_ids_1,attention_mask_1)    # get embeddings for first sentence :: smallBERT--> (batch size,768), largeBERT--> (batch size,1024)
        pooled_output_2 = self.forward(input_ids_2,attention_mask_2)    # get embeddings for second sentence

        if self.config.use_cosine_similarity:                                                               # check cosine similarity is used 
            logits = F.cosine_similarity(pooled_output_1, pooled_output_2)                                  # calculate cosine similarity
        else:
            pooled_output = torch.cat([pooled_output_1,pooled_output_2],dim=1)                              # concatinate two embeddings
            if self.config.use_bert_large:                                                                  # check if largeBERT is used
                pooled_output = F.relu(self.similarity_batch_norm(self.similarity_layer(pooled_output)))    # use linear layer, batch normalization and relu function

            logits = self.similarity_layer_out(pooled_output)[:,0]                                          # use linear layer for classification

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


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    #Sentiment analysis
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    #paraphrase detection
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    #semantic textual similarity
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
                'use_attention_layer':args.use_attention_layer,
                'use_linear_layers':args.use_linear_layers,
                'use_cosine_similarity':args.use_cosine_similarity,
                'use_bert_large':args.use_bert_large,
                'use_conv':args.use_conv,
                'device':device,
                'batch_size':args.batch_size,
                'use_contrastive_loss':args.use_contrastive_loss


    }

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    #CHANGED**** @ 31.12.2023   
    finetune = True if args.option == "finetune" else False     # get finetune variable from args.option
    if args.use_pcgrad:                                         # check if PCGad is used 
        assert  args.use_multitask                              # check if multitasks is used 
        optimizer = PCGrad(optimizer)                           # create PCGrad object
    #CHANGED****           

    #CHANGED**** @ 01.01.2024
    if args.use_triplet_loss:                    # check triplet loss is used 
        assert not args.use_multitask            # check multitask is used
        all_ids_sst = []                         # create list for ids
        all_masks_sst = []                       # create list for masks
        all_labels_sst = []                      # create list for labels
        
        for batch_sst in sst_train_dataloader:                      # get all batch for sst dataset
            all_ids_sst.extend(batch_sst['token_ids'])              # add ids into the list
            all_masks_sst.extend(batch_sst['attention_mask'])       # add masks into the list
            all_labels_sst.extend(batch_sst['labels'])              # add labels into the list
        
        max_ids_len = max(len(ids) for ids in all_ids_sst)                                  # get maximum length of ids
        padded_ids_sst = torch.zeros((len(all_ids_sst), max_ids_len), dtype=torch.int64)    # create zeros tensor for padding ids (some sentence embeddins is less than others, and their shape should be equaled)
        for i, ids in enumerate(all_ids_sst):        # get all ids in sst dataset
            padded_ids_sst[i, :len(ids)] = ids       # add ids into the padded tensor
        all_ids_sst = padded_ids_sst                 
    
        max_masks_len = max(len(masks) for masks in all_masks_sst)                                  # get max length for masks in sst dataset
        padded_masks_sst = torch.zeros((len(all_masks_sst), max_masks_len), dtype=torch.float32)    # # create zeros tensor for padding masks (some sentence embeddins is less than others, and their shape should be equaled)
        for i, masks in enumerate(all_masks_sst):                                                   # get all masks in sst
            padded_masks_sst[i, :len(masks)] = masks                                                # add masks into the padded tensor
        all_masks_sst = padded_masks_sst

    
        #triplet_dataset = TripletDataset(all_ids_sst,all_masks_sst,all_labels_sst,device)          # create triplet dataset

        triplet_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size*4, collate_fn=sst_train_data.collate_fn) # create triplet dataloader object for sst dataset
        triplet_dataloader = iter(triplet_dataloader)       # convert to an iterable object
        model.__setdatasetloader__(triplet_dataloader)      # set dataloader to model
        model.__setssttraindata__(sst_train_data)           # set data to model
        
        triplet_loss = TripletLoss()                        # create triplet loss
    #CHANGED****
    
    
    #CHANGED**** @ 26.12.2023
    best_dev_acc_sst = 0    # initialize variable
    best_dev_acc_para = 0   # initialize variable
    best_dev_acc_sts = 0    # initialize variable
    #CHANGED****

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        #CHANGED**** @ 19.12.2023
        train_dataloaders = [sst_train_dataloader, para_train_dataloader, sts_train_dataloader]                                # add the dataloaders into a list

        for batch_sst, batch_para, batch_sts in tqdm(zip(*train_dataloaders), desc=f'train-{epoch}', disable=TQDM_DISABLE):    # get all batches
            #CHANGED***
                
            b_ids_sst, b_mask_sst, b_labels_sst = (batch_sst['token_ids'],
                                       batch_sst['attention_mask'], batch_sst['labels'])
            #CHANGED**** @ 19.12.2023
            if args.use_multitask:                                                                  # check multitask is used
                (b_ids1_para, b_mask1_para,
                 b_ids2_para, b_mask2_para,
                 b_labels_para) = (batch_para['token_ids_1'], batch_para['attention_mask_1'],
                              batch_para['token_ids_2'], batch_para['attention_mask_2'],
                              batch_para['labels'])                                                 # get ids, masks, labels for paraphrase detection
    
                (b_ids1_sts, b_mask1_sts,
                 b_ids2_sts, b_mask2_sts,
                 b_labels_sts) = (batch_sts['token_ids_1'], batch_sts['attention_mask_1'],
                              batch_sts['token_ids_2'], batch_sts['attention_mask_2'],
                              batch_sts['labels'])                                                  # get ids, masks, labels, for sts task
            #CHANGED****
            
            b_ids_sst = b_ids_sst.to(device)
            b_mask_sst = b_mask_sst.to(device)
            b_labels_sst = b_labels_sst.to(device)
            
            if args.use_multitask:
                #CHANGED**** @ 19.12.2023
                b_ids1_para = b_ids1_para.to(device)            # move the tensors to GPU or CPU
                b_mask1_para = b_mask1_para.to(device)          # move the tensors to GPU or CPU
                b_ids2_para = b_ids2_para.to(device)            # move the tensors to GPU or CPU
                b_mask2_para = b_mask2_para.to(device)          # move the tensors to GPU or CPU
                b_labels_para = b_labels_para.to(device)        # move the tensors to GPU or CPU
    
                b_ids1_sts = b_ids1_sts.to(device)              # move the tensors to GPU or CPU
                b_mask1_sts = b_mask1_sts.to(device)            # move the tensors to GPU or CPU
                b_ids2_sts = b_ids2_sts.to(device)              # move the tensors to GPU or CPU
                b_mask2_sts = b_mask2_sts.to(device)            # move the tensors to GPU or CPU
                b_labels_sts = b_labels_sts.to(device)          # move the tensors to GPU or CPU
                #CHANGED****


            optimizer.zero_grad()
            logits_sst = model.predict_sentiment(b_ids_sst, b_mask_sst)
            
            #CHANGED**** @ 01.01.2024
            # NOTE: the below lines were used to give class weights for each class. However, they did not improved the performance.
            #b_labels_sst_cpu = b_labels_sst.cpu() 

            #class_weights_sst=class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(b_labels_sst_cpu),y=b_labels_sst_cpu.numpy())
            #class_weights_sst_zeros=torch.zeros((5,),dtype=torch.float).to(device)
            #for i,id in enumerate(np.unique(b_labels_sst_cpu)):
            #    class_weights_sst_zeros[id] = class_weights_sst[i] 
            #loss_sst = F.cross_entropy(logits_sst, b_labels_sst.view(-1), weight=class_weights_sst_zeros, reduction='sum') / args.batch_size
            #CHANGED****
            
            #CHANGED**** @ 05.01.2023
            if not args.use_triplet_loss:                                                                         # check triplet loss is used
                loss_sst = F.cross_entropy(logits_sst, b_labels_sst.view(-1), reduction='sum') / args.batch_size
            else:       
                #loss_sst = triplet_loss.forward(logits_sst, b_labels_sst)                                        # use to calculate triplet loss via triplet_loss object 
                loss_sst = model.triplet_loss_sentiment(b_ids_sst, b_mask_sst, b_labels_sst,margin=args.triplet_loss_margin,is_mean=True)       # calculate triplet loss            
            #CHANGED****
            
            
            #CHANGED**** @ 01.01.2024
            # NOTE: These lines are used for contrastive loss. The contrastive loss already was used with triplet loss. Please check use_triplet_loss and use_contrastive_loss parameters
            #loss_sst += model.contrastive_loss_sentiment(b_ids_sst,b_mask_sst,b_labels_sst,all_ids_sst,all_masks_sst,all_labels_sst,device)    # use contrastive loss for multitask learning
            #loss_sst = model.contrastive_loss_sentiment(b_ids_sst,b_mask_sst,b_labels_sst,all_ids_sst,all_masks_sst,all_labels_sst,device)     # use contrastive loss for sst task
            #CHANGED****
            if args.use_multitask:
            
                logits_para = model.predict_paraphrase(b_ids1_para, b_mask1_para, b_ids2_para, b_mask2_para)
        
                #CHANGED**** @ 01.01.2024
                # NOTE: the below lines were used to give class weights for each class. However, they did not improved the performance.
                #b_labels_para_cpu = b_labels_para.cpu()
                #class_weights_para=class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(b_labels_para_cpu),y=b_labels_para_cpu.numpy())
                #class_weights_para_zero=torch.zeros((args.batch_size,),dtype=torch.float).to(device)
                #for i, w in enumerate(class_weights_para):
                #    class_weights_para_zero[b_labels_para_cpu == np.unique(b_labels_para_cpu)[i]] = class_weights_para[i]
                #loss_para = F.cross_entropy(logits_para, b_labels_para.view(-1).float(),weight=class_weights_para_zero, reduction='sum') / args.batch_size
                #CHANGED****

                loss_para = F.cross_entropy(logits_para, b_labels_para.view(-1).float(), reduction='sum') / args.batch_size

                #CHANGED**** @ 01.01.2024
                #loss_para += model.contrastive_loss_sentiment(b_ids_para,b_mask_para,b_labels_para,all_ids_para,all_masks_para,all_labels_para,device) # add loss with contrastive loss
                #CHANGED****

                logits_sts = model.predict_similarity(b_ids1_sts, b_mask1_sts, b_ids2_sts, b_mask2_sts) # get sts logits
                b_labels_sts = b_labels_sts.to(torch.float32)                                           # convert the labels to float32
                loss_sts = F.mse_loss(logits_sts, b_labels_sts.view(-1)) / args.batch_size              # calculate mean square loss

            #CHANGED**** @ 31.12.2023           
                total_loss = loss_sst + loss_para + loss_sts    # sum all losses
            else:
                total_loss = loss_sst                           # just use sst loss            
            #CHANGED****            

            #CHANGED**** @ 31.12.2023
            if args.use_pcgrad:
                total_loss_ = [loss_sst,loss_para,loss_sts]     # append the loseses into a list
                if finetune:                                    # check if finetune is used
                    optimizer.pc_backward(total_loss_)          # calculate gradient
                else:
                    total_loss.backward()                       # calculate gradient 
            else:
                if finetune:                                    # check if finetune is usd
                    total_loss.retain_grad()                    # save the gradients
                    total_loss.backward()                       # calculate gradient
                    
            optimizer.step()         # optimizer step
            #CHANGED****            

            
            train_loss += total_loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        
        train_acc_sst, train_f1_sst, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc_sst, dev_f1_sst, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        #CHANGED**** @ 04.01.2024
        # NOTE: the below lines are used to evaluate the model for each task at every epoch. This cause slow training.
        #train_acc_para,_, _, train_acc_sst,_, _, train_acc_sts, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        #dev_acc_para,_, _, dev_acc_sst,_, _, dev_acc_sts,_, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        #if (dev_acc_para > best_dev_acc_para
        #    and dev_acc_sst > best_dev_acc_sst
        #    and dev_acc_sts > best_dev_acc_sts):
        if dev_acc_sst > best_dev_acc_sst:
            best_dev_acc_sst = dev_acc_sst
            #best_dev_acc_para = dev_acc_para
            #best_dev_acc_sts = dev_acc_sts
            if not args.dontsave_model:
                save_model(model, optimizer, args, config, args.filepath)
        #train_acc = train_acc_sst + train_acc_para + train_acc_sts
        #dev_acc = dev_acc_sst + dev_acc_para + dev_acc_sts
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc_sst :.3f}, dev acc :: {dev_acc_sst :.3f}")
        #print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, average train acc :: {np.mean(train_acc) :.3f}, average dev acc ::{dev_acc :.3f}, sst train acc :: {train_acc_sst}, para train acc :: {train_acc_para}, sts train acc :: {train_acc_sts}, sst dev acc :: {dev_acc_sst}, para dev acc :: {dev_acc_para}, sts dev acc :: {dev_acc_sts}")
        
        #CHANGED****


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        
        if torch.cuda.is_available():
            saved = torch.load(args.filepath, map_location=torch.device('cuda'))
        else:
            saved = torch.load(args.filepath, map_location=torch.device('cpu'))

        #saved = torch.load(args.filepath)
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
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",default=1e-5)

    #CHANGED**** @ 05.01.2023
    argument_list = ["use_multitask","use_pcgrad","use_attention_layer","use_linear_layers","use_cosine_similarity","use_bert_large","use_triplet_loss","use_contrastive_loss"] # create an argument list
    parser.add_argument("--clip", type=float, default=0.25)                         # add clip argument
    parser.add_argument("--use_multitask", action='store_true')                     # add use multitask argument
    parser.add_argument("--use_pcgrad", action='store_true')                        # add use pcgrad argument
    parser.add_argument("--use_attention_layer", action='store_true')               # add use attention layer argument
    parser.add_argument("--use_linear_layers", action='store_true')                 # add use linear layers argument
    parser.add_argument("--use_cosine_similarity", action='store_true')             # add use cosine similarty argument
    parser.add_argument("--use_bert_large", action='store_true')                    # add use largeBERT argument
    parser.add_argument("--use_triplet_loss", action='store_true')                  # add use triplet loss argument
    parser.add_argument("--use_conv", action='store_true')                          # add use convolutional layers argument
    parser.add_argument("--dontsave_model", action='store_true')                    # add dont save model argument
    parser.add_argument("--justtest", action='store_true')                          # add just test argument
    parser.add_argument("--triplet_loss_margin", type=float, default=1.0)           # add trpşlet loss margin argument
    parser.add_argument("--triplet_loss_use_mean", action='store_true')             # add tirpplet loss use mean argument
    parser.add_argument("--use_contrastive_loss", action='store_true')              # add contrastive loss argument
    #CHANGED****

    args = parser.parse_args()

    check_arguments(argument_list,args)

    return args
#CHANGED**** @ 05.01.2023
def check_arguments(argument_list,args):
    print_ = "The training contains the following settings:\n"  # the printed text
    for param_name in argument_list:                            # for each parameters
        if hasattr(args, param_name):                           # check if there is parameter
            param_value = getattr(args, param_name)             # get parameter if it exists
            if param_value:                                     # check if there is parameter in args     
                print_ += f"{param_name}: {param_value}\n"      # add text print_ variable
    print(print_)  
#CHANGED****
if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    if args.justtest:
        test_model(args)
    else:
        train_multitask(args)
        test_model(args)
