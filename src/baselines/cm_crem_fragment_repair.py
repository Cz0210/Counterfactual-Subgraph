"""Task-local repair of CReM0.2.14 single-cut decomposition for salts.

Uncut spectators are not core fragments. They stay in the original molecule
used by CReM's replacement reaction; no salt stripping or database edit occurs.
"""
import ast
import hashlib
import inspect
import textwrap

# inspect.getsource includes the reviewed last-line comment; AST source-segment
# excludes that comment. Bind the actual inspected representation, not AST text.
SOURCE_SHA = 'cfddc826faacc42f05dde44fdeb1dc5830f04dc92640c224eff815c6200efce0'
OLD = 'components = list(Chem.GetMolFrags(chains, asMols=True))'
NEW = '''components = list(Chem.GetMolFrags(chains, asMols=True))
            components = [part for part in components if any(
                a.GetAtomicNum() == 0 and a.GetAtomMapNum() > 0 for a in part.GetAtoms())]
            if len(components) != 2:
                raise ValueError('Single cut must have exactly two attachment-bearing components')'''


def install(native):
    source = textwrap.dedent(inspect.getsource(native.__fragment_mol)).rstrip()
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA or source.count(OLD) != 1:
        raise ValueError('Unreviewed CReM fragmentation implementation')
    original=native.__fragment_mol
    def fragment_components(mol,radius=3,return_ids=True,keep_stereo=False,protected_ids=None,symmetry_fixes=False):
        maps=[]
        parts=native.Chem.GetMolFrags(mol,asMols=True,fragsMolAtomMapping=maps)
        if len(parts)==1:
            return original(mol,radius=radius,return_ids=return_ids,keep_stereo=keep_stereo,
                protected_ids=protected_ids,symmetry_fixes=symmetry_fixes)
        result=[]
        for part,mapping in zip(parts,maps):
            protected=[i for i,old in enumerate(mapping) if old in set(protected_ids or [])]
            # Native MMPA must not cut multiple disconnected components at once:
            # core=None can otherwise contain four attachment-bearing pieces.
            rows=original(part,radius=radius,return_ids=return_ids,keep_stereo=keep_stereo,
                protected_ids=protected,symmetry_fixes=symmetry_fixes)
            for env,core,ids in rows:
                result.append((env,core,tuple(sorted(mapping[i] for i in ids))))
        return result
    native.__fragment_mol = fragment_components
    repaired = source+'\n'+textwrap.dedent(inspect.getsource(fragment_components))
    return {'repair': 'CONNECTED_COMPONENT_NATIVE_FRAGMENTATION_GLOBAL_ATOM_IDS',
            'source_sha256': SOURCE_SHA,
            'repaired_source_sha256': hashlib.sha256(repaired.encode()).hexdigest(),
            'original_full_molecule_retained': True, 'database_changed': False}
