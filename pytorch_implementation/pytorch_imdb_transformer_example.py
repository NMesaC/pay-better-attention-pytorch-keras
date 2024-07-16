"""
## "You Need to Pay Better Attention" PyTorch Transformer Example

## Paper Link: https://arxiv.org/abs/2403.01643

## Author: Nicholas Mesa-Cucalon (https://github.com/NMesaC)
"""
import torch
import time
import os

from torch import nn
from tqdm import tqdm
from IMDB_loader import get_dataloader
from prettytable import PrettyTable


"""
## Setup
"""
# NOTE: Select the shared or per head MultiHeadAttention module for Super Attention
SUPER_SHARED = True
if SUPER_SHARED:
    from pytorch_attention_shared_wa import MultiHeadAttention
else:
    from pytorch_attention_indiv_wa import MultiHeadAttention

# Set device since some classes need info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
## Transformer Block Module
"""
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, layer_type = 'SDPA', max_len = 32, dropout_rate=0.1):
        super().__init__()
        d_k, d_v = d_model // num_heads
        self.att = MultiHeadAttention(num_heads, d_model, d_k, d_v, max_len, layer_type)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

"""
## Embedding Layer
"""
class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos_emb = nn.Embedding(num_embeddings=maxlen, embedding_dim=embed_dim)

    def forward(self, x):
        positions = torch.arange(self.maxlen, device=x.device)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions.unsqueeze(0)
    
"""
## Transformer-Encoder-Only Arch
"""
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_lin, max_len, layer_type, drop_p):
        super().__init__()
        self.multi      = MultiHeadAttention(n_heads,d_model,d_k,d_v,max_len,layer_type)
        self.ff         = nn.Sequential(
                            nn.Linear(d_model,d_lin),
                            nn.ReLU(),
                            nn.Linear(d_lin,d_model)
                         )
        self.norm_multi = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_ff    = nn.LayerNorm(d_model, eps=1e-6)
        self.drop_multi = nn.Dropout(drop_p)
        self.drop_ff    = nn.Dropout(drop_p)
    def forward(self, inp):
        multi = self.multi(inp,inp,inp)
        multi = self.drop_multi(multi)
        z     = self.norm_multi(inp + multi)
        ff    = self.ff(z)
        ff    = self.drop_ff(ff)
        return self.norm_ff(z + ff)

class Encoder(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_lin, n_layers, vocab_size, max_len, layer_type, drop_p):
        super().__init__()
        self.encoder_layers  = nn.Sequential()
        self.embedding       = TokenAndPositionEmbedding(max_len,vocab_size,d_model)
        self.n_layers        = n_layers
        for i in range(n_layers):
            self.encoder_layers.add_module("Encoder_Layer"+str(i),EncoderLayer(n_heads,
                                                                               d_model,
                                                                               d_k,
                                                                               d_v,
                                                                               d_lin,
                                                                               max_len,
                                                                               layer_type,
                                                                               drop_p))
    def forward(self, inp):
        embed_i = self.embedding(inp)
        for layer in self.encoder_layers:
            embed_i     = layer(embed_i)
        return embed_i

class Transformer(nn.Module):
    def __init__(self, 
                 vocab_size = 20000, 
                 n_heads    = 4, 
                 d_model    = 32, 
                 d_k        = 8, 
                 d_v        = 8, 
                 d_lin      = 32, 
                 n_layers   = 1, 
                 max_len    = 32, 
                 layer_type = 'SDPA',
                 drop_p     = 0.1):
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_k
        self.encoder         = Encoder(n_heads,d_model,d_k,d_v,d_lin,n_layers,vocab_size,max_len,layer_type,drop_p)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1        = nn.Dropout(0.1)
        self.dense1          = nn.Linear(d_model, 6)
        self.dropout2        = nn.Dropout(0.1)
        self.dense2          = nn.Linear(6, 1)

    def forward(self, inp):
        x = self.encoder(inp)
        x = x.transpose(1,2)
        x = self.global_avg_pool(x).squeeze(2)
        x = self.dropout1(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout2(x)
        res = self.dense2(x)
        return res

"""
## Training Loop
"""
def train_loop(device, model, optim, loader, loss_func, epoch, train=False):
    # Set training mode
    model.train(mode=train)
    total_loss    = 0
    total_correct = 0
    n_samples     = 0
    label         = 'Training' if train else 'Test'
    for reviews, _, labels in tqdm(loader, desc=f'{label} Epoch: {epoch}'):
        reviews = reviews.to(device)
        labels = labels.to(device)
        #Forward
        if train:
            optim.zero_grad()
            logits = model(reviews)
            loss = loss_func(logits, labels.reshape(logits.shape))
            loss.backward()
            optim.step()
        else:
            with torch.no_grad():
                logits = model(reviews)
                loss   = loss_func(logits, labels.reshape(logits.shape))
        #Predictions
        preds = (logits > 0.5).float()
        #Compute accuracy
        acc = torch.sum(preds == labels.reshape(logits.shape))
        #Track stats
        total_loss += reviews.shape[0] * loss
        n_samples += reviews.shape[0]
        total_correct += acc
    return total_loss / n_samples, total_correct / n_samples

"""
## Checkpoint Callbacks
"""
def save_checkpoint(model, optimizer, epoch, best_metric, filename):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
    return model, optimizer, epoch, best_metric

"""
## Parameter Counting Function
"""
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ("attention" in name.lower()) or ("w_o" in name.lower()):
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    return total_params

def main():
    # Initialize hyperparameters
    vocab_size        = 20000
    batch_size        = 64
    d_model           = 32
    ff_dim            = 32
    max_len           = 32
    num_epochs        = 10
    num_runs          = 5
    n_heads           = 4
    n_layers          = 1
    drop_p            = 0.1
    val_split         = 0.1
    layers            = ['SDPA','Optimised', 'Efficient', 'Super']
    
    # Load data
    train_dataloader, val_dataloader, test_dataloader = get_dataloader("./IMDB_Dataset.csv",
                                                                       vocab_size, 
                                                                       max_len, 
                                                                       batch_size,
                                                                       val_split)
    for layer_type in layers:
        avg_train_loss = 0
        avg_train_acc  = 0
        avg_test_loss  = 0
        avg_test_acc   = 0
        avg_model_size = 0
        run_times      = []
        for _ in range(num_runs):
            # Initialize model, optimizer, and criterion and train/test the model
            print(f"Working with layer type: {layer_type}")
            model_name  = 'best_model.pth'
            transformer = Transformer(vocab_size=vocab_size,
                                      n_heads=n_heads,
                                      n_layers=n_layers,
                                      d_model=d_model,
                                      d_lin=ff_dim,
                                      max_len=max_len,
                                      drop_p = drop_p,
                                      layer_type = layer_type)
            transformer = transformer.to(device)
            #Setup loss function and optimizer
            loss_func = nn.BCEWithLogitsLoss()
            #Performs slightly differently than Keras optimizer
            optim = torch.optim.Adam(transformer.parameters(),lr=1e-3) 
            start_time       = time.time()
            best_val_acc     = -float('inf')
            for epoch in range(num_epochs):
                train_loss, train_acc = train_loop(device,transformer,optim,train_dataloader,loss_func,epoch,True)
                val_loss, val_acc     = train_loop(device,transformer,optim,val_dataloader,loss_func,epoch,False)
                #Print Results per epoch
                print(f" Epoch {epoch}: Train loss: {round(train_loss.item(), 4)} |  Train acc: {round(train_acc.item(), 4)} | \
                Val loss: {round(val_loss.item(), 4)} | Val acc: {round(val_acc.item(), 4)}")
                #Check if our model improved
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(transformer, optim, epoch, best_val_acc, model_name)
            end_time = time.time()
            # Check the best models performance
            best_model, best_optim, start_epoch, _ = load_checkpoint(transformer, optim, model_name)
            test_loss, test_acc   = train_loop(device,best_model,best_optim,test_dataloader,loss_func,start_epoch,False)
            print(f"Best Model: Test Acc {test_acc} | Test Loss {test_loss} \n")
            # Check the size of the best model
            param_size  = 0
            for param in best_model.parameters():
                param_size  += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in best_model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_all_mb = (param_size + buffer_size) / (1024**2)
            # Count number of parameters
            num_params = count_parameters(transformer)
            # Accumulate results
            avg_train_loss += train_loss
            avg_train_acc  += train_acc
            avg_test_loss  += test_loss
            avg_test_acc   += test_acc
            avg_model_size += size_all_mb
            run_times.append(end_time - start_time)
        run_times.sort()
        med_run_time = run_times[len(run_times) // 2]
        file_name = f"{layer_type}_final_results.txt"
        f = open(file_name,"a")
        f.write(f"Average Train Acc over {num_runs} for {layer_type}: {avg_train_acc / num_runs} \n")
        f.write(f"Average Train Loss over {num_runs} for {layer_type}: {avg_train_loss / num_runs} \n")
        f.write(f"Average Test Acc over {num_runs} for {layer_type}: {avg_test_acc / num_runs} \n")
        f.write(f"Average Test Loss over {num_runs} for {layer_type}: {avg_test_loss / num_runs} \n")
        f.write(f"Average Model Size over {num_runs} for {layer_type}: {avg_model_size / num_runs} \n")
        f.write(f"Number of parameters: {num_params} \n")
        f.write(f"Median Run Time over {num_runs} for {layer_type}: {med_run_time} \n")
        f.write("\n")
        f.close()
        
if __name__ == '__main__':
    main()
