import random
import torch
import numpy as np
import tqdm

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test_evaluation(model, test_loader, device):
  loss_fn = torch.nn.CrossEntropyLoss()
  acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())


  model.eval()
  with torch.no_grad():
            losses = []
            acces = []
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'average test loss is {avg_val_loss}, average test acc is {avg_val_acc}.')  
