"""Original-RF fingerprint-environment occlusion; explicitly not Grad-CAM."""
from __future__ import annotations
import hashlib
import numpy as np


def environment_features(smiles, *, radius, bits, original_features):
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms()!=mol.GetNumHeavyAtoms():
        raise ValueError('RF_CM_EXPLICIT_H_OR_INVALID_ATOM_MAPPING')
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=bits)
    additional=rdFingerprintGenerator.AdditionalOutput();additional.AllocateBitInfoMap()
    fp=generator.GetFingerprint(mol,additionalOutput=additional)
    array=np.zeros(bits,dtype=np.float32);DataStructs.ConvertToNumpyArray(fp,array)
    if array.shape!=original_features.shape or not np.array_equal(array,original_features):
        raise ValueError('RF_FEATURE_SCHEMA_OR_ORIGINAL_VECTOR_MISMATCH_NO_FALLBACK')
    info=additional.GetBitInfoMap();atom_bits=[set() for _ in mol.GetAtoms()];environments={}
    for bit,occurrences in info.items():
        environments[str(bit)]=[]
        for center,r in occurrences:
            atoms={int(center)}
            for bond_id in Chem.FindAtomEnvironmentOfRadiusN(mol,int(r),int(center)):
                b=mol.GetBondWithIdx(bond_id);atoms.update((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))
            for atom in atoms:atom_bits[atom].add(int(bit))
            environments[str(bit)].append({'center':int(center),'radius':int(r),'atoms':sorted(atoms)})
    return mol,array,[sorted(x) for x in atom_bits],environments


def attribute(oracle, parent):
    from src.rewards.reward_calculator import smiles_to_morgan_array
    from .cm_crem_generation import make_parent_request
    smiles=parent['smiles']
    original=smiles_to_morgan_array(smiles,radius=oracle.radius,n_bits=oracle.n_bits,clean_dummy_atoms=True)
    if original is None or getattr(oracle.model,'n_features_in_',None)!=oracle.n_bits:
        raise ValueError('RF_AUXILIARY_FEATURES_OR_MISSING_ORIGINAL_VECTOR_REQUIRE_EXPLICIT_MAPPING')
    mol,x,mapping,environments=environment_features(smiles,radius=oracle.radius,bits=oracle.n_bits,original_features=original)
    empty=x.copy();baseline=oracle.model.predict_proba(x[None])[0]
    if not np.array_equal(baseline,oracle.model.predict_proba(empty[None])[0]) or not np.array_equal(baseline,oracle.predict_proba([smiles])[0]):
        raise ValueError('RF_ORIGINAL_OR_EMPTY_OCCLUSION_PREDICTION_CHANGED')
    queries=np.repeat(x[None],len(mapping),axis=0)
    for atom,features in enumerate(mapping):queries[atom,features]=0
    proba=np.asarray(oracle.model.predict_proba(queries),dtype=np.float64)
    scores=baseline[oracle.source_label]-proba[:,oracle.source_label]
    if not np.isfinite(scores).all():raise ValueError('RF_NONFINITE_ATTRIBUTION')
    selected=sorted(range(len(mapping)),key=lambda i:(-scores[i],i))[:max(1,len(mapping)//5)]
    return {'parent_id':parent['parent_id'],'split':'train','smiles':smiles,
        'method':'RF-FeatureOcclusion','not_gradcam':True,'oracle_sha':oracle.checkpoint_id,
        'before_probability':baseline.tolist(),'before_label':int(oracle.class_labels[int(np.argmax(baseline))]),
        'original_vector_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'dtype':'float32',
        'fingerprint':{'radius':oracle.radius,'bits':oracle.n_bits,'useChirality':False,'count_mode':False,'aux_features':False},
        'atom_feature_indices':mapping,'environments':environments,
        'collision_bits':[bit for bit,rows in environments.items() if len(rows)>1],
        'scores':scores.tolist(),'occluded_source_probabilities':proba[:,oracle.source_label].tolist(),
        'generation_request':make_parent_request(parent['parent_id'],mol,selected),
        'interpretation':'Feature occlusion is mask guidance, not atom-level causal attribution; verify generated molecules with original RF.'}
