"""Bounded exact theta-close counting with reusable embeddings and packed masks."""
from __future__ import annotations
import json
import os
from pathlib import Path
import resource

from .rf_aligned_pool import atomic_json, digest, file_sha


def rss_bytes():
    # This CPU production entrypoint runs on Linux Slurm (ru_maxrss is KiB).
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def atomic_npz(path: Path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def decode_mask(packed, shape):
    import numpy as np
    return np.unpackbits(packed, bitorder="little", count=int(shape[0]) * int(shape[1])).reshape(shape).astype(bool)


def count_exact_pairs(*, graphs, all_sources, source_positions, model, element_counts,
                      identity, root: Path, max_rss_bytes: int, batch_size: int = 128):
    import numpy as np
    import torch
    from torch_geometric.data import Batch
    root.mkdir(parents=True, exist_ok=True)
    identity_sha = digest(identity)
    source_path = root / "sources.npz"
    source_receipt = root / "sources.json"
    if source_receipt.exists():
        proof = json.loads(source_receipt.read_text())
        if proof["identity_sha"] != identity_sha or file_sha(source_path) != proof["sha256"]:
            raise ValueError("Source embedding cache identity mismatch")
        with np.load(source_path, allow_pickle=False) as values:
            embeddings = torch.from_numpy(values["embeddings"].copy())
            counts = torch.from_numpy(values["counts"].copy())
    else:
        with torch.no_grad():
            all_embeddings = model.embed_model(Batch.from_data_list(all_sources)).detach().cpu()
        embeddings = all_embeddings[source_positions]
        counts = element_counts(all_sources).cpu()[source_positions]
        atomic_npz(source_path, embeddings=embeddings.numpy(), counts=counts.numpy())
        atomic_json(source_receipt, {"identity_sha": identity_sha, "sha256": file_sha(source_path), "source_positions": source_positions, "all_sources_count": len(all_sources), "rss_bytes": rss_bytes()})
    chunks = []
    for chunk_index, begin in enumerate(range(0, len(graphs), batch_size)):
        stop = min(len(graphs), begin + batch_size)
        values_path = root / f"chunk-{chunk_index:06d}.npz"
        receipt_path = root / f"chunk-{chunk_index:06d}.json"
        if receipt_path.exists():
            proof = json.loads(receipt_path.read_text())
            if proof["identity_sha"] != identity_sha or proof["begin"] != begin or proof["stop"] != stop or file_sha(values_path) != proof["sha256"]:
                raise ValueError("Exact count checkpoint mismatch")
        else:
            chunk = graphs[begin:stop]
            with torch.no_grad():
                candidate_embeddings = model.embed_model(Batch.from_data_list(chunk)).detach().cpu()
                candidate_counts = element_counts(chunk).cpu()
                scale = candidate_counts[:, None] + counts[None, :]
                distances = torch.cdist(candidate_embeddings, embeddings, p=2) / scale
                mask = (distances <= .1).numpy()
            packed = np.packbits(mask.reshape(-1), bitorder="little")
            if not np.array_equal(decode_mask(packed, mask.shape), mask):
                raise ValueError("Packed exact theta mask failed round-trip")
            atomic_npz(values_path, embeddings=candidate_embeddings.numpy(), counts=candidate_counts.numpy(), mask=packed)
            locations = np.argwhere(mask)
            first = None if not len(locations) else {"local_candidate": int(locations[0, 0]), "local_parent": int(locations[0, 1]), "normalized_greed_distance": float(distances[locations[0, 0], locations[0, 1]])}
            proof = {"identity_sha": identity_sha, "begin": begin, "stop": stop, "path": str(values_path), "sha256": file_sha(values_path), "mask_shape": list(mask.shape), "pair_count": int(mask.sum()), "first_close_pair": first, "rss_bytes": rss_bytes()}
            atomic_json(receipt_path, proof)
        if rss_bytes() > max_rss_bytes:
            raise RuntimeError("Count-only stage exceeded its RSS guard")
        chunks.append(proof)
        atomic_json(root / "progress.json", {"state": "COUNT_ONLY", "candidates_completed": stop, "candidate_count": len(graphs), "exact_close_pairs_so_far": sum(x["pair_count"] for x in chunks), "full_pair_vectors_created": False, "rss_bytes": rss_bytes()})
    result = {"state": "COUNT_COMPLETE", "identity_sha": identity_sha, "pair_count": sum(x["pair_count"] for x in chunks), "cartesian_upper_bound": len(graphs) * len(source_positions), "vector_dim": int(embeddings.shape[1]), "source_cache": str(source_path), "chunks": chunks, "full_pair_vectors_created": False, "embeddings_reused_by_materialization": True, "mask_is_exact": True}
    atomic_json(root / "manifest.json", result)
    return result, embeddings, counts
