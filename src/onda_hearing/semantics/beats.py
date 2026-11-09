"""Onda Sound Brain: optional BEATs AudioSet classifier backend.

Defaults to user's specified ONDA_BEATS_CHECKPOINT path. This is an integration
name, NOT proof that the referenced Microsoft BEATs cpt2 weights have been
further fine-tuned by Onda. For custom fine-tuned heads, provide your own
SoundBrain adapter rather than silently loading incompatible checkpoint heads.
"""
from __future__ import annotations
import importlib, json, os, sys
from pathlib import Path
import numpy as np
from .core import EventScore

DEFAULT_CHECKPOINT='$HOME/Developer/onda-foundation/models/pretrained/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'

class OndaSoundBrain:
    name='Onda Sound Brain'
    def __init__(self,checkpoint_path:str|None=None,labels_path:str|None=None,
                 beats_module_dir:str|None=None,device:str|None=None,top_k:int=5,
                 min_score:float=.15,allow_unsafe_checkpoint:bool=False):
        """Torch checkpoint loading is disabled unless explicitly trusted.

        BEATs uses Python pickle checkpoints; arbitrary untrusted checkpoints
        can be dangerous. `allow_unsafe_checkpoint` is an explicit opt-in.
        No implicit downloading, no automatic fallback and no model on the
        audio producer or policy thread.
        """
        import torch
        self.checkpoint=Path(os.path.expandvars(checkpoint_path or os.getenv('ONDA_BEATS_CHECKPOINT',DEFAULT_CHECKPOINT))).expanduser()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'Onda Sound Brain checkpoint not found: {self.checkpoint}')
        if not allow_unsafe_checkpoint:
            raise PermissionError('BEATs checkpoint is a pickle file: set allow_unsafe_checkpoint=True only for a trusted local checkpoint')
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.top_k=int(top_k);self.min_score=float(min_score)
        if self.top_k<1 or not 0<=self.min_score<=1:raise ValueError('invalid top-k / threshold')
        search_dir=beats_module_dir or os.environ.get('ONDA_BEATS_MODULE_DIR','')
        if search_dir:
            module_dir=str(Path(search_dir).expanduser().resolve())
            if module_dir not in sys.path:sys.path.insert(0,module_dir)
        try:
            mod=importlib.import_module('BEATs')
            BEATs=mod.BEATs; BEATsConfig=mod.BEATsConfig
        except (ImportError,AttributeError) as exc:
            raise ImportError('BEATs.py not importable; set ONDA_BEATS_MODULE_DIR to trusted microsoft/unilm/beats source') from exc
        # Only load your own/trusted checkpoint. Strict state-dict validation.
        try:
            ckpt=torch.load(str(self.checkpoint),map_location='cpu',weights_only=True)
        except Exception as exc:
            raise RuntimeError('safe BEATs checkpoint load failed; inspect the checkpoint rather than enabling general pickle execution') from exc
        if not isinstance(ckpt,dict) or 'model' not in ckpt or 'cfg' not in ckpt:
            raise ValueError('expected BEATs checkpoint with model and cfg')
        self.model=BEATs(BEATsConfig(ckpt['cfg']))
        self.model.load_state_dict(ckpt['model'],strict=True)
        self.model.to(self.device).eval()
        label_file=labels_path or os.getenv('ONDA_BEATS_LABELS')
        if not label_file:
            raise ValueError('set ONDA_BEATS_LABELS to explicit AudioSet ordered 527-label mapping')
        with Path(label_file).expanduser().open(encoding='utf8') as f: loaded=json.load(f)
        if isinstance(loaded,list):
            self.labels=[str(x['display_name'] if isinstance(x,dict) and 'display_name' in x else x) for x in loaded]
        elif isinstance(loaded,dict) and all(k.isdigit() for k in loaded):
            self.labels=[str(loaded[str(i)]) for i in range(len(loaded))]
        else:
            raise ValueError('labels must be ordered list or contiguous integer-key dictionary')
        if len(self.labels)!=527:
            raise ValueError(f'expected 527 AudioSet labels, got {len(self.labels)}')

    def classify(self,mono:np.ndarray,sample_rate:int)->tuple[EventScore,...]:
        import torch
        if sample_rate!=16000:raise ValueError('Onda Sound Brain requires 16 kHz mono PCM')
        x=np.asarray(mono,dtype=np.float32)
        if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('finite mono PCM required')
        tensor=torch.from_numpy(np.array(x, dtype=np.float32, copy=True, order='C')).unsqueeze(0).to(self.device)
        padding_mask=torch.zeros_like(tensor,dtype=torch.bool)
        with torch.inference_mode():
            output=self.model.extract_features(tensor,padding_mask=padding_mask)
        logits=output[0] if isinstance(output,(list,tuple)) else output
        if not hasattr(logits,'shape') or logits.ndim!=2 or logits.shape[0]!=1 or logits.shape[1]!=len(self.labels):
            raise RuntimeError(f'unexpected BEATs predictions shape: {getattr(logits,"shape",None)}')
        # Official microsoft/unilm BEATs.extract_features() applies sigmoid for fine-tuned classifiers.
        # Do not apply sigmoid twice (that would corrupt probabilities).
        scores=logits.float()[0].detach().cpu().numpy()
        if not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):
            raise RuntimeError('BEATs returned invalid probability scores')
        inds=np.argsort(-scores,kind='stable')[:self.top_k]
        return tuple(EventScore(self.labels[int(i)],float(scores[i])) for i in inds if scores[i]>=self.min_score)
