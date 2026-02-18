# src/train/train_3dcnn.py
import os,csv,random
import numpy as np
import torch,torch.nn as nn,torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision.models.video import r2plus1d_18,R2Plus1D_18_Weights

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
MANIFEST=os.path.join(ROOT,"data_processed","manifest.csv")
MODEL_PATH=os.path.join(ROOT,"models","best.pt")

device="cuda" if torch.cuda.is_available() else "cpu"

class ClipDS(Dataset):
    def __init__(self,rows,labels):
        self.rows=rows
        self.l2i={l:i for i,l in enumerate(labels)}

    def __len__(self): return len(self.rows)

    def __getitem__(self,i):
        p,l=self.rows[i]
        x=np.load(p)["x"]
        x=torch.from_numpy(x).permute(3,0,1,2)
        return x.float(),torch.tensor(self.l2i[l])

def load_data():
    with open(MANIFEST) as f:
        r=list(csv.reader(f))[1:]
    labels=sorted(set(x[1] for x in r))
    random.shuffle(r)
    split=int(len(r)*0.8)
    return r[:split],r[split:],labels

def main():
    tr,va,labels=load_data()
    trdl=DataLoader(ClipDS(tr,labels),4,True)
    vadl=DataLoader(ClipDS(va,labels),4)

    model=r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    model.fc=nn.Linear(model.fc.in_features,len(labels))
    model=model.to(device)

    opt=optim.Adam(model.parameters(),lr=1e-4)
    ce=nn.CrossEntropyLoss()

    best=0
    for ep in range(20):
        model.train()
        for x,y in trdl:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            loss=ce(model(x),y)
            loss.backward()
            opt.step()

        # val
        model.eval();correct=0;total=0
        with torch.no_grad():
            for x,y in vadl:
                x,y=x.to(device),y.to(device)
                pred=model(x).argmax(1)
                correct+= (pred==y).sum().item()
                total+=y.size(0)

        acc=correct/total
        print("epoch",ep,"val_acc",acc)

        if acc>best:
            best=acc
            torch.save({"model":model.state_dict(),"labels":labels},MODEL_PATH)

    print("BEST",best)

if __name__=="__main__":
    main()
