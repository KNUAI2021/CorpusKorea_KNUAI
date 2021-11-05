"""
CorpusKorea
TEAM KNUAI

python evaluation.py --task_name BoolQ --device cuda:1 --state_file 4_BoolQ_KNUAI_v3_model_save --test_file ./corpus/SKT_BoolQ_Test.tsv --output_dir ./json

"""
from transformers import AdamW
from transformers import ElectraTokenizer, ElectraModel, ElectraForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch

import os
import argparse
import json
import pandas as pd

class COPAmodel(nn.Module):
    def __init__(self, electra, num_labels, nhead, num_layers):
        super(COPAmodel, self).__init__()
        self.num_labels = num_labels
        self.nhead = nhead
        self.num_layers = num_layers
        self.electra = electra
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = 768, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = num_layers)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, self.num_labels)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, sent_ids, token_type_ids, attention_mask):
        
        cls_hs = self.electra(sent_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:,0] # use seg
        #cls_hs = self.electra(sent_ids, attention_mask = mask)[0][:,0] # not use seg
        
        cls_hs = cls_hs.unsqueeze(dim=0)
    
        x = self.decoder(cls_hs, cls_hs)
        
        #x = x.view(-1, 768)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        #print(x.shape)
        return x


class TaskEvaluator:
    def __init__(self, args) -> None:
        self.MAX_LEN = 64
        self.args = args
        print(self.args)
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        if self.args.task_name == "copa":
            self.model = COPAmodel(ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator"), num_labels=2, nhead=8, num_layers=6)
        else:
            self.model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",num_labels=2)
        self.model.to(args.device)
        print(f"-----{self.args.task_name} model loaded-----")
        
        self.state = torch.load(os.path.join(self.args.state_dir, self.args.state_file))
        self.model.load_state_dict(self.state['model_state_dict'])
        # self.model.load_state_dict(torch.load(self.args.state_file))


    def pad(self, data, pad_id, max_len):
        padded_data = list(map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]), data))
        return padded_data

    def read_data_for_test(self):
        self.raw_data = pd.read_csv(os.path.join(self.args.test_file_dir, self.args.test_file), sep='\t', header=0)

    def evaluate(self):
        test_inputs = self.read_data_for_test()
        test_dataset = TensorDataset(test_inputs, test_inputs)
        test_dataloader = DataLoader(test_dataset, sampler=None, batch_size=self.args.batch_size)

        print(f"-----{self.args.task_name} Evaluation start-----")
        self.model.eval()
        all_logits = []
        for batch in test_dataloader:
            with torch.no_grad():
                data, _ = batch
                src = data[:, 0, :]
                segs = data[:, 1, :]
                mask = data[:, 2, :]
                
                src = src.to(self.args.device)
                segs = segs.to(self.args.device)
                mask = mask.to(self.args.device)
                    
                # Forward
                outputs = self.model(src, 
                                token_type_ids=segs, 
                                attention_mask=mask)
                # loss
                logits = outputs[0]
                all_logits.append(logits)


        all_logits = torch.cat(all_logits, dim=0)
        if self.args.task_name == "copa":
            all_logits = all_logits.view(500, -1)
        probs = F.softmax(all_logits, dim=1).cpu()
        probs = probs.max(-1)[1]
        if self.args.task_name == "copa":
            probs = torch.where(probs==2,1,probs)
            probs = torch.where(probs==0,2,probs)
            probs = torch.where(probs==3,2,probs)        

        probs = probs.tolist()


        file_path = os.path.join(self.args.output_dir, f"{self.args.task_name}.json")

        data = {}

        data[self.args.task_name] = []

        for i in range(len(test_inputs)):
            if self.args.task_name == 'wic':
                probs[i] = True if probs[i] else False
                data[self.args.task_name].append({"idx": i, "label": probs[i]})
            elif self.args.task_name == 'copa':
                if i >= len(test_inputs)//2:
                    break
                data[self.args.task_name].append({"idx": i, "label": probs[i]})
            elif self.args.task_name == 'cola':
                data[self.args.task_name].append({"idx": i, "label": probs[i]})
            elif self.args.task_name == 'boolq':
                data[self.args.task_name].append({"idx": i, "label": probs[i]})

        return data


class CoLAEvaluator(TaskEvaluator):
    def __init__(self, args) -> None:
        super().__init__(args)

    
    def read_data_for_test(self):
        super().read_data_for_test()
        raw_data = self.raw_data
        inputs = []
        segs = []
        
        
        sentences = raw_data['sentence'].values.tolist()

        for i in range(len(raw_data)):
            input_ids1 = torch.tensor(self.tokenizer.encode(sentences[i], add_special_tokens=True))
            inputs.append(input_ids1) 
            
            clss = torch.cat([torch.where(input_ids1 == 2)[0], torch.tensor([len(input_ids1)])])
            
            for n, val in enumerate(clss[:-1]):
                if n % 2 == 0:
                    res1 = torch.LongTensor(list([0] * (clss[n + 1] - clss[n])))
            segs.append(res1)
                    
        input_tensor = self.pad(inputs, 0, self.MAX_LEN)
        seg_tensor = self.pad(segs, 0, self.MAX_LEN)
        
        input_tensor = torch.stack(input_tensor, dim=0)
        seg_tensor = torch.stack(seg_tensor, dim=0)
        mask_tensor = (~ (input_tensor == 0))
        
        output_tensor = torch.cat([input_tensor.unsqueeze(dim=1) , seg_tensor.unsqueeze(dim=1), mask_tensor.unsqueeze(dim=1)], dim=1)

        return output_tensor


class WiCEvaluator(TaskEvaluator):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.MAX_LEN = 256

    def read_data_for_test(self):
        super().read_data_for_test()
        raw_data = self.raw_data
        inputs = []
        segs = []
        sentence1 = raw_data['SENTENCE1'].values.tolist()
        sentence2 = raw_data['SENTENCE2'].values.tolist()  

        for i in range(len(raw_data)):
            input_ids1 = torch.tensor(self.tokenizer.encode(sentence1[i], add_special_tokens=True)).unsqueeze(0)
            input_ids2 = torch.tensor(self.tokenizer.encode(sentence2[i], add_special_tokens=True)).unsqueeze(0)
            input_cat = torch.cat([input_ids1, input_ids2],-1)
            input_cat = input_cat.squeeze()
            inputs.append(input_cat)
            
            clss = torch.cat([torch.where(input_cat == 2)[0], torch.tensor([len(input_cat)])])
            
            for n, val in enumerate(clss[:-1]):
                if n % 2 == 0:
                    res1 = torch.LongTensor(list([0] * (clss[n + 1] - clss[n])))
                else:
                    res2 = torch.LongTensor(list([1] * (clss[n + 1] - clss[n])))
                
                        
            seg_cat = torch.cat([res1.unsqueeze(0), res2.unsqueeze(0)], -1)
            segs.append(seg_cat.squeeze())
            
        input_tensor = self.pad(inputs, 0, self.MAX_LEN)
        seg_tensor = self.pad(segs, 0, self.MAX_LEN)
        
        input_tensor = torch.stack(input_tensor, dim=0)
        seg_tensor = torch.stack(seg_tensor, dim=0)
        mask_tensor = (~ (input_tensor == 0))\
        
        output_tensor = torch.cat([input_tensor.unsqueeze(dim=1) , seg_tensor.unsqueeze(dim=1), mask_tensor.unsqueeze(dim=1)], dim=1)
        

        return output_tensor


class COPAEvaluator(TaskEvaluator):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.MAX_LEN = 128
    
    def read_data_for_test(self):
            super().read_data_for_test()
            raw_data = self.raw_data
            inputs = []
            segs = []
            sentence = raw_data['sentence'].values.tolist()
            question = raw_data['question'].values.tolist()
            choice1 = raw_data['1'].values.tolist()
            choice2 = raw_data['2'].values.tolist()
            answer = raw_data['Answer'].values.tolist()

            for i in range(len(raw_data)):
                input_ids1 = torch.tensor(self.tokenizer.encode(sentence[i] +' '+ question[i], add_special_tokens=True)).unsqueeze(0)
                input_ids2 = torch.tensor(self.tokenizer.encode(choice1[i], add_special_tokens=True)).unsqueeze(0)
                input_cat = torch.cat([input_ids1, input_ids2],-1)
                input_cat = input_cat.squeeze()
                inputs.append(input_cat) #########
                
                clss = torch.cat([torch.where(input_cat == 2)[0], torch.tensor([len(input_cat)])])
                #print(clss)
                
                #seg = list()
                for n, val in enumerate(clss[:-1]):
                    if n % 2 == 0:
                        res1 = torch.LongTensor(list([0] * (clss[n + 1] - clss[n])))
                    else:
                        res2 = torch.LongTensor(list([1] * (clss[n + 1] - clss[n])))
                    
                            
                seg_cat = torch.cat([res1.unsqueeze(0), res2.unsqueeze(0)], -1)
                segs.append(seg_cat.squeeze())
                        
                input_ids1 = torch.tensor(self.tokenizer.encode(sentence[i] +' '+ question[i], add_special_tokens=True)).unsqueeze(0)
                input_ids2 = torch.tensor(self.tokenizer.encode(choice2[i], add_special_tokens=True)).unsqueeze(0)
                input_cat = torch.cat([input_ids1, input_ids2],-1)
                input_cat = input_cat.squeeze()
                inputs.append(input_cat) #########
                
                clss = torch.cat([torch.where(input_cat == 2)[0], torch.tensor([len(input_cat)])])
                #print(clss)
                
                #seg = list()
                for n, val in enumerate(clss[:-1]):
                    if n % 2 == 0:
                        res1 = torch.LongTensor(list([0] * (clss[n + 1] - clss[n])))
                    else:
                        res2 = torch.LongTensor(list([1] * (clss[n + 1] - clss[n])))
                    
                            
                seg_cat = torch.cat([res1.unsqueeze(0), res2.unsqueeze(0)], -1)
                segs.append(seg_cat.squeeze())
                
            
            input_tensor = self.pad(inputs, 0, self.MAX_LEN)
            seg_tensor = self.pad(segs, 0, self.MAX_LEN)
            
            input_tensor = torch.stack(input_tensor, dim=0)
            seg_tensor = torch.stack(seg_tensor, dim=0)
            mask_tensor = (~ (input_tensor == 0))\
            
            output_tensor = torch.cat([input_tensor.unsqueeze(dim=1) , seg_tensor.unsqueeze(dim=1), mask_tensor.unsqueeze(dim=1)], dim=1)
            

            return output_tensor


class BoolQEvaluator(TaskEvaluator):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.MAX_LEN = 512

    def read_data_for_test(self):
        super().read_data_for_test()
        raw_data = self.raw_data
        inputs = []
        segs = []
        text = raw_data['Text'].values.tolist()
        question = raw_data['Question'].values.tolist()

        for i in range(len(raw_data)):
            input_ids1 = torch.tensor(self.tokenizer.encode(text[i], add_special_tokens=True)).unsqueeze(0)
            input_ids2 = torch.tensor(self.tokenizer.encode(question[i], add_special_tokens=True)).unsqueeze(0)
            input_cat = torch.cat([input_ids1, input_ids2],-1)
            input_cat = input_cat.squeeze()
            inputs.append(input_cat) #########
            
            clss = torch.cat([torch.where(input_cat == 2)[0], torch.tensor([len(input_cat)])])
            
            for n, val in enumerate(clss[:-1]):
                if n % 2 == 0:
                    res1 = torch.LongTensor(list([0] * (clss[n + 1] - clss[n])))
                else:
                    res2 = torch.LongTensor(list([1] * (clss[n + 1] - clss[n])))
                
                        
            seg_cat = torch.cat([res1.unsqueeze(0), res2.unsqueeze(0)], -1)
            segs.append(seg_cat.squeeze())
            
        input_tensor = self.pad(inputs, 0, self.MAX_LEN)
        seg_tensor = self.pad(segs, 0, self.MAX_LEN)
        
        input_tensor = torch.stack(input_tensor, dim=0)
        seg_tensor = torch.stack(seg_tensor, dim=0)
        mask_tensor = (~ (input_tensor == 0))\
        
        output_tensor = torch.cat([input_tensor.unsqueeze(dim=1) , seg_tensor.unsqueeze(dim=1), mask_tensor.unsqueeze(dim=1)], dim=1)
    

        return output_tensor
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--device", type=str, help="device info")
    parser.add_argument("--state_dir", type=str, help="state file directory for tasks")
    parser.add_argument("--test_file_dir", type=str, help="test file directory for tasks")
    parser.add_argument("--output_dir", type=str, default='./', help="json output dir")

    parser.add_argument("--task_name", type=str, help="task name. CoLA|WiC|COPA|BoolQ")
    parser.add_argument("--state_file", type=str, help="model state for (CoLA|WiC|COPA|BoolQ) task")
    parser.add_argument("--test_file", type=str, help="test file for (CoLA|WiC|COPA|BoolQ) task")
    args = parser.parse_args()
    args.device = torch.device(args.device)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    tasks = {
        "CoLA": {"Evaluator": CoLAEvaluator},
        "WiC": {"Evaluator": WiCEvaluator},
        "COPA": {"Evaluator": COPAEvaluator},
        "BoolQ": {"Evaluator": BoolQEvaluator}
    }
    states_list = os.listdir(args.state_dir)
    test_file_list = os.listdir(args.test_file_dir)
    
    for task in tasks.keys():
        for state in states_list:
            if state.lower().startswith(task.lower()):
                tasks[task]['state_file'] = state 
                break
        
        for test_file in test_file_list:
            if task in test_file or task.lower() in test_file:
                tasks[task]['test_file'] = test_file
                
    with open(os.path.join(args.output_dir, 'KNUAI_v3_result.json'), 'w') as outfile:
        result = {}            
        for key, val in tasks.items():
                args.task_name, args.state_file, args.test_file = key.lower(), val['state_file'], val['test_file']
                result.update(val["Evaluator"](args).evaluate())
        json.dump(result, outfile, indent=4)


if __name__ == "__main__":
    main()