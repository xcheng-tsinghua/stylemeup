"""Native PyTorch first-order meta-training for the released StyleMeUp VAE."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader
from PIL import Image
from Networks import VAE_Network, Regulariser
from qmul_data import QMUL, image_tensor, raster
from checkpoint_writer import AsyncCheckpointWriter


class Objective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = VAE_Network()

    def forward(self, images):
        p, n, s1, s2 = images.unbind(0)
        inputs = (p, n, s1, s2)
        encoded = [self.vae.encoder(x.unsqueeze(0)) for x in inputs]
        inv = [e[1] for e in encoded]
        z = [e[0].rsample() for e in encoded]
        rec = [self.vae.decoder(a+b) for a, b in zip(inv, z)]
        kl = torch.stack([.5*(e[0].mean.square()+e[0].variance-1-e[0].variance.log()).mean() for e in encoded]).mean()
        self_rec = sum(F.mse_loss(r, x.unsqueeze(0)) for r, x in zip(rec, inputs))/4
        cross_style = (F.mse_loss(self.vae.decoder(inv[2]+z[3]), s2.unsqueeze(0)) + F.mse_loss(self.vae.decoder(inv[3]+z[2]), s1.unsqueeze(0)))/2
        cross_domain = sum(F.mse_loss(rec[i], p.unsqueeze(0)) for i in (2,3))/4
        cross_domain += sum(F.mse_loss(rec[0], x.unsqueeze(0)) for x in (s1,s2))/4
        triplet = F.triplet_margin_loss(torch.cat(inv[2:]), inv[0].expand(2,-1), inv[1].expand(2,-1), margin=.2)
        combined = [a+b for a,b in zip(inv,z)]
        triplet += F.triplet_margin_loss(torch.cat(combined[2:]), combined[0].expand(2,-1), combined[1].expand(2,-1), margin=.2)
        # Include self reconstruction, which was computed but omitted in main.py.
        return kl + self_rec + cross_style + cross_domain + triplet


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size, output):
    model.eval()
    def encode(paths, sketch=False):
        features = []
        for start in range(0, len(paths), batch_size):
            images = [image_tensor(raster(p) if sketch else Image.open(p)) for p in paths[start:start+batch_size]]
            features.append(model.vae.encoder(torch.stack(images).to(device))[1].cpu())
        return torch.cat(features)
    gallery = encode([dataset.photos[i] for i in dataset.ids])
    queries = encode(dataset.sketches, True)
    distances = torch.cdist(queries, gallery)
    indices = {name:i for i,name in enumerate(dataset.ids)}
    targets = torch.tensor([indices[p.stem.rsplit('_',1)[0]] for p in dataset.sketches])
    target_dist = distances[torch.arange(len(targets)), targets]
    ranks = (distances <= target_dist[:,None]).sum(1)
    metrics = dict(queries=len(queries), gallery=len(gallery), top1=(ranks<=1).float().mean().item(),
                   top5=(ranks<=5).float().mean().item(), top10=(ranks<=10).float().mean().item(),
                   mrr=(1/ranks.float()).mean().item(), mean_rank=ranks.float().mean().item())
    rows = [dict(sketch=p.stem, target=p.stem.rsplit('_',1)[0], rank=int(ranks[k]),
                 top10=[dataset.ids[j] for j in distances[k].argsort()[:10].tolist()]) for k,p in enumerate(dataset.sketches)]
    Path(output).write_text(json.dumps(dict(metrics=metrics, retrievals=rows), indent=2), encoding='utf-8')
    return metrics


def main(argv=None, tracker=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'D:\document\DataSet\sketch_retrieval\qmul_v2_fit')
    ap.add_argument('--category', choices=['shoe','chair'], required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--meta-batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--fast-lr', type=float, default=.005)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--eval-batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-steps', type=int, default=0, help='Smoke test only; 0 processes all episodes')
    ap.add_argument('--resume')
    ap.add_argument('--evaluate-only', action='store_true')
    args = ap.parse_args(argv)
    if args.epochs < 1 or args.meta_batch < 1 or args.eval_batch < 1:
        ap.error('epochs and batch sizes must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / f'last_{args.seed}.pt'
    if ((out/'last.pt').exists() or any(out.glob('last_*.pt'))) and not args.resume:
        raise FileExistsError('Output contains a checkpoint; use --resume or a new --output')
    train = QMUL(args.root,args.category,'train'); test = QMUL(args.root,args.category,'test')
    manifest = dict(config=vars(args), train=train.audit(), test=test.audit(), torch=torch.__version__,
                    device=str(device), gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                    protocol='FG-SBIR, disjoint instances, test-instance gallery, all test sketches')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest), flush=True)
    model = Objective().to(device); reg = Regulariser().to(device)
    optimizer = torch.optim.Adam(list(model.parameters())+list(reg.parameters()), lr=args.lr)
    first_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume,map_location=device,weights_only=False)
        if ckpt['config']['category'] != args.category:
            raise ValueError('Checkpoint category does not match --category')
        model.load_state_dict(ckpt['model']); reg.load_state_dict(ckpt['regulariser'])
        optimizer.load_state_dict(ckpt['optimizer']); first_epoch=ckpt['epoch']+1
        random.setstate(ckpt['python_rng']); np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng']])
    if args.evaluate_only:
        if not args.resume: raise ValueError('--evaluate-only requires --resume')
        print(evaluate(model,test,device,args.eval_batch,out/'evaluation.json'),flush=True); return
    if tracker is not None:
        tracker.start(args, ckpt if args.resume else None)
    loader = DataLoader(train,batch_size=args.meta_batch,shuffle=True,num_workers=args.workers)
    start=time.time()
    with AsyncCheckpointWriter(checkpoint_path) as writer:
        for epoch in range(first_epoch,args.epochs):
            model.train(); losses=[]
            for step,batch in enumerate(loader,1):
                optimizer.zero_grad(set_to_none=True)
                for episode in batch.to(device):
                    styles=random.sample([2,3,4],3)
                    params=dict(model.named_parameters())
                    inner=model(episode[[0,1,styles[0],styles[1]]])
                    grads=torch.autograd.grad(inner,tuple(params.values()))
                    adapted={name:p-args.fast_lr*g.detach() for (name,p),g in zip(params.items(),grads)}
                    # Preserve first-order MAML for VAE weights, but retain the analytic
                    # derivative through the learned weighted-L1 regulariser.
                    for name,scale in [('vae.encoder.invar.weight',reg.params[:,:-1]),('vae.encoder.invar.bias',reg.params[:,-1])]:
                        adapted[name]=adapted[name]-args.fast_lr*scale*params[name].detach().sign()
                    loss=functional_call(model,adapted,(episode[[0,1,styles[1],styles[2]]],))
                    if not torch.isfinite(loss): raise RuntimeError('Non-finite meta loss')
                    (loss/len(batch)).backward(); losses.append(loss.item())
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in reg.parameters()):
                    raise RuntimeError('Missing/nonfinite regulariser hypergradient')
                torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(reg.parameters()),10,error_if_nonfinite=True)
                optimizer.step()
                if step==1 or step%10==0:
                    print(f'epoch={epoch+1} step={step}/{len(loader)} loss={np.mean(losses[-len(batch):]):.5f} elapsed={time.time()-start:.1f}s',flush=True)
                if args.max_steps and step>=args.max_steps: break
            metrics=evaluate(model,test,device,args.eval_batch,out/f'retrieval_epoch_{epoch+1:03}.json')
            record=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.time()-start,**metrics)
            checkpoint=dict(model=model.state_dict(),regulariser=reg.state_dict(),optimizer=optimizer.state_dict(),
                            epoch=epoch,config=vars(args),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
            if tracker is not None:
                checkpoint['swanlab'] = tracker.identity
                checkpoint['record'] = record
            writer.submit(checkpoint)
            if tracker is not None:
                tracker.log(record)
            with (out/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
