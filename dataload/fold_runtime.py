"""折内冻结向量与检索结果的预计算表（不依赖可训练权重）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from graph_llm.aux.prompt_utils import item_meta_from_row
from graph_llm.dataload.dataloader import FEATURE_CORE_WEIGHT, GraphCollater, GraphDataset
from graph_llm.dataload.tail_stats import is_content_token


@dataclass
class FoldRuntimeCache:
    """主进程持有的折级查表缓存。"""

    node_emb_by_token_id: dict[int, torch.Tensor] = field(default_factory=dict)
    item_emb_by_text: dict[str, torch.Tensor] = field(default_factory=dict)
    # (split_name, local_idx) -> prepare_batch 单样本逻辑结果
    review_pick_by_sample: dict[tuple[str, int], dict] = field(default_factory=dict)
    # (split_name, local_idx) -> (target_token_ids, core_feature_token_ids)
    selector_labels_by_sample: dict[tuple[str, int], tuple[frozenset[int], frozenset[int]]] = (
        field(default_factory=dict)
    )


def _collect_graphs(graph_manager, split_name: str, dataset: GraphDataset):
    graphs = []
    for idx in range(len(dataset)):
        graphs.append(graph_manager.get_graph(split_name, idx))
    return graphs


def build_node_token_embedding_table(
    embedding_encoder,
    tokenizer,
    graph_managers: dict[str, object],
    datasets: dict[str, GraphDataset],
    split_names: dict[str, str],
) -> dict[int, torch.Tensor]:
    """扫描折内所有图节点 token，一次性编码 surface embedding。"""
    token_ids: set[int] = set()
    for split_key, dataset in datasets.items():
        split_name = split_names[split_key]
        manager = graph_managers[split_key]
        for graph in _collect_graphs(manager, split_name, dataset):
            if graph.num_nodes == 0:
                continue
            token_ids.update(int(t) for t in graph.node_token_ids.tolist())
    if not token_ids:
        return {}
    ordered = sorted(token_ids)
    decode_fn = lambda tid: tokenizer.decode([int(tid)], skip_special_tokens=True)
    vectors = embedding_encoder.encode_token_ids(ordered, decode_fn)
    table: dict[int, torch.Tensor] = {}
    for token_id, vector in zip(ordered, vectors):
        table[int(token_id)] = vector.detach().float().cpu()
    return table


def build_item_text_embedding_table(
    embedding_encoder,
    datasets: dict[str, GraphDataset],
    item_meta: dict,
) -> dict[str, torch.Tensor]:
    """每个唯一 item_text 只编码一次。"""
    texts: set[str] = set()
    for dataset in datasets.values():
        for idx in range(len(dataset)):
            row = dataset[idx]
            raw_item = str(row["raw_item"]) if "raw_item" in row else str(row["item"])
            _title, _desc, item_text = item_meta_from_row(raw_item, item_meta)
            texts.add(item_text)
    if not texts:
        return {}
    ordered = sorted(texts)
    vectors = embedding_encoder.encode_texts(ordered)
    return {
        text: vectors[pos].detach().float().cpu()
        for pos, text in enumerate(ordered)
    }


def _precompute_review_pick(review_memory, context: dict, item_query_vec: torch.Tensor) -> dict:
    """复用 ReviewMemoryBank 的资格与 topk 规则，单样本预计算。"""
    item_query_vec = item_query_vec.detach().float().cpu()
    user_rows = review_memory._eligible_rows(
        review_memory.user_to_rows.get(str(context["raw_user"]), []),
        context,
    )
    selected_user_rows = review_memory._topk(user_rows, item_query_vec, review_memory.top_k_user)
    user_query = torch.zeros((review_memory.embedding_dim,), dtype=torch.float32)
    if selected_user_rows:
        selected = review_memory.embeddings[selected_user_rows].float()
        user_query = F.normalize(selected.mean(dim=0), dim=-1)

    item_rows = review_memory._eligible_rows(
        review_memory.item_to_rows.get(str(context["raw_item"]), []),
        context,
    )
    item_query = user_query if selected_user_rows else item_query_vec
    selected_item_rows = review_memory._topk(item_rows, item_query, review_memory.top_k_item)
    return {
        "selected_user_rows": selected_user_rows,
        "selected_item_rows": selected_item_rows,
        "user_query": user_query,
    }


def build_review_pick_tables(
    review_memories: dict[str, object],
    datasets: dict[str, GraphDataset],
    item_emb_by_text: dict[str, torch.Tensor],
    item_meta: dict,
) -> dict[tuple[str, int], dict]:
    """为每个样本预计算评论 Top-K 行号（CPU float32 topk，与在线逻辑一致）。"""
    picks: dict[tuple[str, int], dict] = {}
    for split_key, dataset in datasets.items():
        review_memory = review_memories.get(split_key)
        if review_memory is None:
            continue
        for idx in range(len(dataset)):
            row = dataset[idx]
            raw_item = str(row["raw_item"]) if "raw_item" in row else str(row["item"])
            _title, _desc, item_text = item_meta_from_row(raw_item, item_meta)
            item_query = item_emb_by_text.get(item_text)
            if item_query is None:
                continue
            context = {
                "raw_user": str(row["raw_user"]) if "raw_user" in row else str(row["user"]),
                "raw_item": raw_item,
                "split_name": str(row.get("split_name", split_key)),
                "local_idx": int(row["local_idx"]),
            }
            picks[(context["split_name"], int(context["local_idx"]))] = _precompute_review_pick(
                review_memory,
                context,
                item_query,
            )
    return picks


def build_selector_label_tables(
    collaters: dict[str, GraphCollater],
    datasets: dict[str, GraphDataset],
    tokenizer,
    ignored_token_ids: set[int],
) -> dict[tuple[str, int], tuple[frozenset[int], frozenset[int]]]:
    """按样本预计算 selector 正样本与核心 feature token 集合。"""
    labels: dict[tuple[str, int], tuple[frozenset[int], frozenset[int]]] = {}
    for split_key, dataset in datasets.items():
        collater = collaters[split_key]
        for idx in range(len(dataset)):
            row = dataset[idx]
            split_name = str(row.get("split_name", split_key))
            cached = collater._sample_cache.get(idx)
            if cached is None:
                continue
            text_ids = [int(t) for t in cached["text_ids"]]
            feature_weights = cached["feature_weights"]
            target_ids = {
                int(token_id)
                for token_id in text_ids
                if is_content_token(tokenizer, int(token_id), ignored_token_ids)
            }
            core_feature_ids = {
                int(token_id)
                for token_id, weight in zip(text_ids, feature_weights)
                if float(weight) >= FEATURE_CORE_WEIGHT
                and is_content_token(tokenizer, int(token_id), ignored_token_ids)
            }
            labels[(split_name, idx)] = (frozenset(target_ids), frozenset(core_feature_ids))
    return labels


def build_fold_runtime_cache(
    *,
    embedding_encoder,
    tokenizer,
    item_meta: dict,
    graph_managers: dict[str, object],
    datasets: dict[str, GraphDataset],
    split_names: dict[str, str],
    collaters: dict[str, GraphCollater],
    review_memories: dict[str, object],
    ignored_token_ids: set[int],
) -> FoldRuntimeCache:
    """折初始化时构建全部冻结查表。"""
    node_table = build_node_token_embedding_table(
        embedding_encoder,
        tokenizer,
        graph_managers,
        datasets,
        split_names,
    )
    item_table = build_item_text_embedding_table(embedding_encoder, datasets, item_meta)
    review_picks = build_review_pick_tables(review_memories, datasets, item_table, item_meta)
    selector_labels = build_selector_label_tables(collaters, datasets, tokenizer, ignored_token_ids)
    return FoldRuntimeCache(
        node_emb_by_token_id=node_table,
        item_emb_by_text=item_table,
        review_pick_by_sample=review_picks,
        selector_labels_by_sample=selector_labels,
    )
