"""Train-only development cohort and bounded on-demand MolCLR node cache."""
from collections import defaultdict, OrderedDict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import numpy as np
from .ours_taste_focus_matrix import dump_json


def validate_gpu_reservation(registry,index,uuid,observed_uuid):
    assert observed_uuid.strip()==uuid, 'GPU_INDEX_UUID_CHANGED'
    for row in registry['gpu_leases']:
        if row['gpu']==index:
            assert row['state']=='RELEASED', 'PROJECT_FUTURE_GPU_RESERVED'


def stratified_development(rows, limit=256, seed=7):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    assert all(r['pred_before']==1 for r in rows)
    margins=np.array([r['p_before'][1]-max(r['p_before'][0],r['p_before'][2]) for r in rows])
    cuts=np.quantile(margins,[.25,.5,.75])
    groups=defaultdict(list)
    for row,margin in zip(rows,margins,strict=True):
        mol=Chem.MolFromSmiles(row['smiles'])
        assert mol is not None
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=True)
        band=int(np.searchsorted(cuts,margin,side='right'))
        groups[(scaffold,band)].append({**row,'scaffold':scaffold,'margin':float(margin),'margin_band':band})
    def order(value): return hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()
    for values in groups.values(): values.sort(key=lambda r:order(r['parent_id']))
    keys=sorted(groups,key=lambda k:order(str(k)))
    selected=[]; cursor=0
    while len(selected)<min(limit,len(rows)):
        for key in keys:
            if cursor<len(groups[key]): selected.append(groups[key][cursor])
            if len(selected)>=min(limit,len(rows)): break
        cursor+=1
    assert len({r['parent_id'] for r in selected})==len(selected)
    return selected


def compact_embedder_class():
    from .molclr_node_embeddings import MolCLRNodeEmbedder, MoleculeNodeEmbedding, canonicalize_smiles, atom_numbers_for_smiles
    class CompactEmbedder(MolCLRNodeEmbedder):
        """Same numerical encoder; SQLite blobs avoid one file per molecule.

        The pre-existing cache is consulted read-only. Cache misses use the
        unchanged _compute_node_embeddings path and persist in one new DB.
        """
        def __init__(self,*,compact_db,**kwargs):
            super().__init__(**kwargs)
            self.db=sqlite3.connect(str(compact_db))
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS nodes (key TEXT PRIMARY KEY,payload BLOB NOT NULL)')
            self.lru=OrderedDict(); self.lru_bytes=0

        def get(self,smiles):
            canonical=canonicalize_smiles(smiles)
            if canonical is None: raise ValueError('INVALID_NODE_INPUT')
            key=json.dumps(self.cache_payload(canonical),sort_keys=True)
            if key in self.lru:
                self.stats.node_embedding_cache_hits+=1
                self.lru.move_to_end(key); return self.lru[key]
            row=self.db.execute('SELECT payload FROM nodes WHERE key=?',(key,)).fetchone()
            data=None
            if row is not None:
                with np.load(io.BytesIO(row[0]),allow_pickle=False) as z:
                    data=MoleculeNodeEmbedding(canonical,z['H'],z['atoms'])
                self.stats.node_embedding_cache_hits+=1
            if data is None:
                for path in (self.cache_path(canonical),self.legacy_cache_path(canonical)):
                    if path.is_file():
                        data=self._load_npz(path,canonical)
                        self.stats.node_embedding_cache_hits+=1
                        break
            if data is None:
                h=self._compute_node_embeddings(canonical)
                data=MoleculeNodeEmbedding(canonical,h,atom_numbers_for_smiles(canonical))
                self.stats.node_embedding_cache_misses+=1
            assert data.H.ndim==2 and data.H.shape[0]==len(data.atom_numbers) and np.isfinite(data.H).all()
            if row is None:
                pages=self.db.execute('PRAGMA page_count').fetchone()[0]
                page_size=self.db.execute('PRAGMA page_size').fetchone()[0]
                if pages*page_size>2*1024**3: raise RuntimeError('COMPACT_NODE_CACHE_2GIB_BUDGET')
                buf=io.BytesIO();np.savez_compressed(buf,H=data.H,atoms=data.atom_numbers)
                self.db.execute('INSERT INTO nodes VALUES (?,?)',(key,buf.getvalue()))
            n=data.H.nbytes+data.atom_numbers.nbytes
            while self.lru and self.lru_bytes+n>256*1024**2:
                _,old=self.lru.popitem(last=False);self.lru_bytes-=old.H.nbytes+old.atom_numbers.nbytes
            if n<=256*1024**2: self.lru[key]=data;self.lru_bytes+=n
            return data

        def commit(self): self.db.commit()
        def close(self): self.db.commit();self.db.close()
    return CompactEmbedder
