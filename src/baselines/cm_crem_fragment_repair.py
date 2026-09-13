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
    repaired = source.replace(OLD, NEW)
    ast.parse(repaired)
    # Only this fresh worker's imported module is changed. Shared package files
    # and all other workers retain their original implementation.
    namespace = dict(vars(native))
    exec(compile(repaired, '<CM-single-cut-spectator-repair>', 'exec'), namespace)
    native.__fragment_mol = namespace['__fragment_mol']
    return {'repair': 'SINGLE_CUT_ATTACHMENT_COMPONENTS_ONLY',
            'source_sha256': SOURCE_SHA,
            'repaired_source_sha256': hashlib.sha256(repaired.encode()).hexdigest(),
            'original_full_molecule_retained': True, 'database_changed': False}
