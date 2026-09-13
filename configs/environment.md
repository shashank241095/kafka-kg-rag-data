# HippoRAG External Baseline — Environment, Configuration and Source-Level Evidence

Recorded **before** any Kafka retrieval result was observed. Every configuration value
below is sourced to the pinned official implementation, never chosen after inspecting
retrieval performance.

## 1. Pinned implementation

| Field | Value |
|---|---|
| Repository | `https://github.com/OSU-NLP-Group/HippoRAG` |
| Branch | `legacy` (the repo's own designation for the NeurIPS 2024 implementation) |
| Tag | `v1.0.0` |
| Commit SHA | `b144c46df14cabe5f5822d8caded4bec5f709461` |
| Local path | `third_party/hipporag` |
| Checkout state | pristine — `git status --porcelain` empty; `git tag --points-at HEAD` = `v1.0.0` |

Not HippoRAG 2 (`main` branch / `pip install hipporag`). Not a fork.

## 2. Environment

| Component | Version | Source |
|---|---|---|
| Python | 3.10.21 (Homebrew `python@3.10`) | D3 |
| Platform | macOS 15.7.3, arm64, Apple M4 Pro, 24 GB | — |
| torch | 1.13.1 | upstream pin |
| colbert-ai | 0.2.19 | upstream pin |
| transformers | 4.37.2 | upstream pin |
| numpy | 1.26.4 | upstream pin |
| igraph | 0.11.4 | upstream pin |
| scipy | 1.12.0 | upstream pin |
| openai | 1.12.0 | upstream pin |
| faiss | **faiss-cpu 1.8.0** (upstream: `faiss-gpu==1.7.2`) | D1 |
| venv | `.venv-hipporag` | — |
| Requirements actually installed | `scripts/hipporag/requirements-macos-arm64.txt` | derived from upstream, substitutions in header |

Device resolution observed at runtime:

```
torch.cuda.is_available()  -> False
torch.cuda.device_count()  -> 0     => colbert use_gpu = False
torch.backends.mps         -> True  (unused; ColBERT has no MPS path)
colbert.parameters.DEVICE  -> cpu
```

`colbert-ai` 0.2.19 gates every CUDA call on `use_gpu = config.total_visible_gpus > 0`
(`colbert/searcher.py:43-49`, `colbert/modeling/colbert.py:22-35,105,167-198`,
`colbert/search/candidate_generation.py:9-61`, `colbert/search/index_loader.py:14-44`,
`colbert/search/index_storage.py:21-28`), so with zero visible GPUs the library takes its
own supported CPU path. ColBERT also ships `conda_env_cpu.yml` upstream, i.e. CPU-only
operation is an intended configuration, not an improvisation.

### ColBERTv2 checkpoint

| Field | Value |
|---|---|
| URL | `https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz` (the URL in upstream's README) |
| Size | 387 MB |
| SHA256 (tarball) | `d07df4b2a9fe1848fbed333127a1a5b4f2d96106f420948f0db4450b018cb119` |
| Location | `third_party/hipporag/exp/colbertv2.0` (the path upstream requires) |
| `nbits` | 2 (upstream default in `src/colbertv2_indexing.py`) |

## 3. Frozen retrieval universe

| Artifact | SHA256 |
|---|---|
| `artifacts/hipporag/frozen_corpus_1600.jsonl` | `950cf89775df0f4d2a6b5940c0d1246bd1655315481dd4322b12825885351889` |
| `artifacts/hipporag/doc_id_mapping.csv` | `0a1b3ff120dbcd4eeecaf67886c53b96a7ba95479abfa50b1d84f39744f057eb` |
| `artifacts/hipporag/corpus.json` | `f6433eb8c64e9e389c25363487481978bb1a92968f3bd3ccd9edc907e176f21e` |
| `artifacts/hipporag/queries.json` | `eac04364f27778d22ff9fcb082330e15328beff63c391415f0bd75ed15079fc3` |

The snapshot is `chmod 444`. `scripts/hipporag/build_frozen_corpus.py` refuses to overwrite it.
It is the single source of documents for **both** the ColBERTv2-only and the full-HippoRAG
condition; neither re-reads Qdrant or Neo4j.

Document text is byte-identical to the frozen dense baseline's serialization
(`scripts/embed_logs_to_qdrant.py::build_point_text`, copied verbatim into the snapshot
builder):

```
[{LEVEL}] {template} (logger={logger} method={method_fqn} class={class_fqn} location={file}:{line})
```

No tabs, no newlines, no CRs in any of the 1,600 records; all 1,600 texts unique; all 1,600
`(file,line)` targets unique. Stable ids: `doc_id` is `uuid5(NAMESPACE_URL, "{repo}::{file}::{line}")`,
the same id the frozen Qdrant collection uses; `idx` is the 0-based position under the frozen
Cypher's `ORDER BY repo, file, line`, which is also the ColBERT pid.

## 4. Configuration, with provenance

| Parameter | Value | Provenance |
|---|---|---|
| OpenIE / extraction LLM | `gpt-3.5-turbo-1106` | `README.md:121`, `src/setup_hipporag_main_exps.sh`, `src/ircot_hipporag.py:120` default |
| LLM API provider | `openai` | `README.md:124` |
| Retriever (both conditions) | `colbertv2` | `README.md:150-156`, `src/run_hipporag_main_exps.sh` |
| **Synonym threshold** | **0.8** | `README.md:122` `SYNONYM_THRESH=0.8` (see §5) |
| PPR damping | 0.5 | `README.md:156`, `src/run_hipporag_main_exps.sh` |
| `doc_ensemble` | `f` | `README.md:156` |
| `max_steps` | 1 (single-step retrieval) | `README.md:156` |
| `recognition_threshold` | 0.9 | `src/ircot_hipporag.py:124` default |
| `extraction_type` | `ner` | `src/setup_hipporag_colbert.sh` |
| Synonymy edge construction | `--cosine_sim_edges` | `src/setup_hipporag_colbert.sh` |
| `graph_alg` | `ppr` | `src/ircot_hipporag.py:122` default |
| `top_k` | **20** | evaluation cutoff; D5, see §6 |

## 5. Synonym threshold — resolving the 0.5 / 0.8 question

These were two different parameters, not two values of one parameter. Upstream's
`README.md:156` retrieval command reads:

```
--max_steps 1 --doc_ensemble f --top_k 10  --sim_threshold $SYNONYM_THRESH --damping 0.5
```

`--sim_threshold` is the synonym/entity-similarity threshold and `--damping` is the PPR
damping factor. The official custom-dataset instructions set them independently:

- `README.md:117` heading: *"Indexing with ColBERTv2 for Synonymy Edges"*
- `README.md:122`: `SYNONYM_THRESH=0.8`
- `README.md:126`: `bash src/setup_hipporag_colbert.sh $DATA $LLM $GPUS $SYNONYM_THRESH $LLM_API`
- `README.md:156`: `--sim_threshold $SYNONYM_THRESH --damping 0.5`

The README introduces this block with *"We will use the best hyperparameters defined in our
paper"*. The ColBERTv2 custom-dataset path therefore prescribes **synonym threshold = 0.8**,
and the same 0.8 is used in `src/setup_hipporag_main_exps.sh` (`syn_threshold=0.8`) for the
paper's own main experiments.

**Chosen: `sim_threshold = 0.8`, `damping = 0.5`.** Recorded before any Kafka retrieval
result existed. Not to be tuned afterward.

## 6. `top_k` semantics — proof it is truncation only

**Every** occurrence of `top_k` in `src/hipporag.py`:

```
173:    def rank_docs(self, query: str, top_k=10):          # signature
177:        @param top_k: the number of documents to return  # docstring
294:        return sorted_doc_ids.tolist()[:top_k], sorted_scores.tolist()[:top_k], logs
```

Three occurrences: signature, docstring, return. The ranking itself is computed over the
whole corpus before `top_k` is applied:

```
291:        sorted_doc_ids = np.argsort(doc_prob, kind='mergesort')[::-1]
292:        sorted_scores = doc_prob[sorted_doc_ids]
...
294:        return sorted_doc_ids.tolist()[:top_k], sorted_scores.tolist()[:top_k], logs
```

`top_k` reaches neither the ColBERT search breadth, the entity linking, the graph, nor PPR:

- ColBERT search breadth is the full corpus, independent of `top_k` —
  `hipporag.py:195` (`k=self.doc_to_phrases_mat.shape[0]`) and `hipporag.py:196`
  (`k=len(self.dataset_df)`).
- The PPR block (`hipporag.py:212-235`) reads `all_phrase_weights`, `self.graph_alg`,
  `self.damping`; never `top_k`.
- `link_node_by_colbertv2` / `link_node_by_dpr` take `query_ner_list` only.

In `src/ircot_hipporag.py`, `top_k` is passed to `retrieve_step` (`:224`), used in the output
filename (`:169,173`), and used at `:233` inside `while it < max_steps:` (`:230`) — which with
`--max_steps 1` never executes.

**Conclusion: `top_k` controls only final returned-ranking truncation → `top_k=20` is used.**
Because the truncation is provably pure, the runner additionally records the gold's rank in
the untruncated corpus-wide ranking, which changes no computation.

## 7. `dpr_only=True` semantics — proof this is the controlled retriever floor

Why ColBERTv2-only is the right floor: it is HippoRAG's *own* backbone with the graph
switched off, executed through the same harness, the same checkpoint, the same index and the
same document text. The difference between the two conditions is therefore the graph
machinery alone — not a change of retrieval model. This is also the pairing the HippoRAG
paper itself reports (a `ColBERTv2` baseline alongside `HippoRAG (ColBERTv2)`).

**Every** occurrence of `dpr_only` in `src/hipporag.py`:

```
 33:                 colbert_config=None, dpr_only=False, graph_alg='ppr', damping=0.1, ...
 46:        @param dpr_only: Flag to determine whether HippoRAG will be used at all
 97:        self.dpr_only = dpr_only
102:        if not self.dpr_only:
113:        if (doc_ensemble or dpr_only) and self.linking_retriever_name not in ['colbertv2', 'bm25']:
118:            if self.dpr_only is False or self.doc_ensemble:
123:            if self.doc_ensemble or dpr_only:
193:            elif self.dpr_only:
203:            if self.doc_ensemble or self.dpr_only:
212:        if not self.dpr_only:
232:            else:  # dpr_only or no entities found
236:        if self.doc_ensemble or self.dpr_only:
267:        if not (self.dpr_only) and len(query_ner_list) > 0:
297:        if self.dpr_only:
```

What each guard establishes:

**(a) No query NER — `hipporag.py:296-299`**
```python
def query_ner(self, query):
    if self.dpr_only:
        query_ner_list = []
```
No entities are extracted from the query, so nothing can be linked into the graph.

**(b) No graph is loaded or built — `hipporag.py:102-107`**
```python
if not self.dpr_only:
    self.load_index_files()
    # Construct Graph
    self.build_graph()
    # Loading Node Embeddings
    self.load_node_vectors()
else:
    self.load_corpus()
```
Under `dpr_only` the OpenIE triples, the phrase dictionary, the synonymy edges and the
node embeddings are never read; only the corpus is loaded.

**(c) No phrase index — `hipporag.py:118-122`**
```python
if self.dpr_only is False or self.doc_ensemble:
    colbertv2_index(self.phrases.tolist(), ... 'phrase' ...)
```
The entity/phrase ColBERT index is not built. Only the document index is
(`hipporag.py:123-127`, `if self.doc_ensemble or dpr_only:`).

**(d) Direct ColBERTv2 corpus search — `hipporag.py:193-197`**
```python
elif self.dpr_only:
    query_doc_scores = np.zeros(len(self.dataset_df))
    ranking = self.corpus_searcher.search_all(queries, k=len(self.dataset_df))
    for doc_id, rank, score in ranking.data[0]:
        query_doc_scores[doc_id] = score
```
Every one of the 1,600 documents is scored by ColBERTv2 late interaction.

**(e) No PPR — `hipporag.py:212`**
```python
if not self.dpr_only:
    ...
    ppr_phrase_probs = self.run_pagerank_igraph_chunk([all_phrase_weights])[0]
```
`run_pagerank_igraph_chunk` is unreachable; `damping` is never consulted.

**(f) Ranking is the raw ColBERTv2 score order — `hipporag.py:236-240`, `:291-294`**
```python
if self.doc_ensemble or self.dpr_only:
    if len(query_ner_list) == 0:
        doc_prob = query_doc_scores
...
sorted_doc_ids = np.argsort(doc_prob, kind='mergesort')[::-1]
```
`query_ner_list` is `[]` by (a), so the first branch always taken: `doc_prob` is exactly the
ColBERTv2 score vector. No ensembling, no `recognition_threshold` blend.

### How the control is implemented

For the ColBERTv2-only control, we use HippoRAG's upstream ColBERT indexing and search
components with a custom-data adapter that preserves the frozen benchmark document text
exactly. The stock v1 custom-loading path introduces an additional delimiter artifact during
TSV serialization; this is treated as a data-loading compatibility issue rather than part of
the retrieval algorithm.

Concretely, `scripts/hipporag/run_retriever_only.py` transcribes (d) and (f) verbatim, builds
the index with upstream's unmodified `src/colbertv2_indexing.py::colbertv2_index`, and
constructs upstream's `Searcher` exactly as `hipporag.py:123-127`. No BGE reranker, no AST
edges, no `text-embedding-3-large`, and zero LLM API calls in this condition. The delimiter
artifact and a supplementary sensitivity run through the stock loading path are documented in
§13.

## 8. OpenIE model availability

Probe of the pinned model (1 completion token) on 2026-09-09:

```
POST https://api.openai.com/v1/chat/completions
{"model":"gpt-3.5-turbo-1106","messages":[{"role":"user","content":"ok"}],"max_tokens":1}
-> 200; "model": "gpt-3.5-turbo-1106"; finish_reason "length"; 9 total tokens
   id chatcmpl-EMLODBiRpbwti6EcaKawJIimzdVsH; system_fingerprint fp_6ca79bc10d
```

**The pinned OpenIE model serves.** No substitution is required and none is made; Phase 1C
does not trigger. (Note: `kg/python/.env` sets `OPENAI_MODEL=gpt-4o-mini` for other scripts
in this project. That variable is not used by the HippoRAG path, which takes the model from
the pinned `gpt-3.5-turbo-1106` argument.)

## 9. Deviations from the published configuration

| # | Deviation | Category | Intended algorithmic change to ranking / graph / ColBERT / extraction / synonym edges / PPR? |
|---|---|---|---|
| D1 | `faiss-gpu==1.7.2` → `faiss-cpu==1.8.0`; dropped four `nvidia-*-cu11` wheels | platform | None intended. Same k-means/IVF procedure on CPU; minor platform-dependent numerical differences remain possible. |
| D2 | ColBERTv2 runs on CPU (`total_visible_gpus=0`) | platform | None intended. The library's own gated CPU path; minor platform-dependent numerical differences remain possible. |
| D3 | Python 3.10.21 instead of 3.9 | platform | None intended. `colbert-ai` declares `>=3.8`. |
| D4 | Corpus/query loading via an adapter in `scripts/hipporag/`; two `os.makedirs` calls for directories upstream assumes exist | data plumbing | None intended. Zero edits to `third_party/hipporag`; checkout verified pristine. See §13. |
| D5 | `top_k=20` instead of the README's `10` | evaluation cutoff | None — proven in §6 to be return-statement truncation only. |
| D6 | ColBERT's `Launcher.launch` redirected to its own `run_process_without_mp` at `nranks==1`; runner bodies placed under an `if __name__ == "__main__"` guard | platform | None intended. See §11. |
| D7 | `langchain-community` 0.0.38 → 0.2.5, `langchain` 0.1.20 → 0.2.17, `langchain-core` 0.1.53 → 0.2.43, `langchain-openai` 0.1.5 → 0.1.25, `langchain-together` 0.1.2 → 0.1.5 | dependency resolution | None intended. See §12. |

No substantive hyperparameter is altered: `sim_threshold=0.8`, `damping=0.5`,
`doc_ensemble=f`, `max_steps=1`, `recognition_threshold=0.9`, `extraction_type=ner`,
`graph_alg=ppr`, `nbits=2`, `--cosine_sim_edges` are all at their official values.

## 10. Determinism

| Item | Value |
|---|---|
| Corpus order | frozen Cypher `ORDER BY repo, file, line`; `idx` = 0..1599 |
| Query order | frozen benchmark file order |
| Tie-breaking in ranking | `np.argsort(..., kind='mergesort')` — stable, as upstream |
| Bootstrap seed (evaluation) | 42, 10,000 replicates, resampling the 103 target clusters — matching the frozen analysis |

## 11. D6 — ColBERT multiprocessing on Apple Silicon

Two independent failures had to be resolved before ColBERTv2 would index at all on this
machine. Both are process-topology issues; neither touches any computation.

**(a) `Launcher.launch` deadlock.** `colbert/infra/launcher.py:11` forces
`mp.set_start_method('spawn', force=True)`. `Launcher.launch` then starts one `mp.Process`
per rank and blocks on `return_value_queue.get()`. On macOS arm64 the spawned rank-0 child
sat at 0% CPU indefinitely — reproduced over 12 minutes with no index bytes written and only
3.7s of CPU consumed.

ColBERT already ships the single-process alternative and reaches it upstream via
`avoid_fork_if_possible`:

```
colbert/indexer.py:86      if self.config.nranks == 1 and self.config.avoid_fork_if_possible:
colbert/indexer.py:89          launcher.launch_without_fork(...)
colbert/infra/launcher.py:84   def launch_without_fork(self, custom_config, *args):
colbert/infra/launcher.py:90       assert (custom_config.avoid_fork_if_possible or self.run_config.avoid_fork_if_possible)
colbert/infra/launcher.py:93       return run_process_without_mp(self.callee, new_config, *args)
```

Upstream HippoRAG's `colbertv2_index()` builds its own `ColBERTConfig(nbits=2, root=...)`
without that flag, so the flag cannot be set without editing `third_party/hipporag`. The
adapter therefore redirects the launcher in-process instead:

```python
def _launch_single_process(self, custom_config, *args):
    if self.nranks != 1:
        return _launch_orig(self, custom_config, *args)
    new_config = type(custom_config).from_existing(custom_config, self.run_config, _cl.RunConfig(rank=0))
    return _cl.run_process_without_mp(self.callee, new_config, *args)
```

`run_process_without_mp` and `setup_new_process` are the same computation
(`colbert/infra/launcher.py:105-127`): both call `set_seed(12345)` and then
`with Run().context(config, inherit_config=False): callee(config, *args)`. `setup_new_process`
additionally calls `distributed.init()` and posts to an `mp.Queue` — neither is meaningful at
`nranks == 1`. No intended algorithmic change to embeddings, centroids, residuals, scores or
ranking; minor platform-dependent numerical differences remain possible.

**(b) `mp.Manager()` re-import.** `colbert/indexer.py:93` calls `mp.Manager()` *before* the
launch. Under the forced `spawn` method the manager child re-imports `__main__`, so any
runner whose work sits at module level is re-executed in the child and dies with
`_check_not_importing_main` → `EOFError` in the parent. Every script here therefore puts its
work in `main()` behind `if __name__ == "__main__"`. Observed and fixed in
`scripts/hipporag/crosscheck_dpr_only.py`.

**(c) `ninja` on `PATH`.** `torch.utils.cpp_extension.verify_ninja_availability()` looks for
the `ninja` *executable*, not the installed Python package, so ColBERT's
`segmented_maxsim_cpp` / `segmented_lookup_cpp` extensions fail to build unless
`.venv-hipporag/bin` is on `PATH`. The runner is invoked with that prefix.

Also observed (harmless): `torch/amp/autocast_mode.py` warns
*"User provided device_type of 'cuda', but CUDA is not available. Disabling"* — ColBERT
requests autocast unconditionally and torch disables it. Encoding proceeds in fp32.

## 12. D7 — langchain dependency resolution

Upstream `requirements.txt:138-141` lists `langchain`, `langchain-openai`,
`langchain-together` and `langchain_community` **with no version pins**. A clean install
today resolves `langchain-community==0.0.38`, under which upstream's own code does not
import:

```
src/named_entity_extraction_parallel.py:7   from langchain_community.chat_models import ChatOllama, ChatLlamaCpp
src/openie_with_retrieval_option_parallel.py:5  (same import)
src/langchain_util.py:41                    from langchain_community.chat_models import ChatLlamaCpp
ImportError: cannot import name 'ChatLlamaCpp' from 'langchain_community.chat_models'
```

`ChatLlamaCpp` first appears in `langchain-community 0.2.5` (absent in 0.2.4), which requires
`langchain-core>=0.2.7`, which in turn forces `langchain` and `langchain-openai` forward. The
**minimum** set that makes upstream importable was chosen:

| Package | Resolved-from-unpinned | Installed | Reason |
|---|---|---|---|
| `langchain-community` | 0.0.38 | **0.2.5** | first version exporting `ChatLlamaCpp` |
| `langchain-core` | 0.1.53 | **0.2.43** | required by community 0.2.5 (`>=0.2.7,<0.3.0`) |
| `langchain` | 0.1.20 | **0.2.17** | 0.1.20 requires `core<0.2` |
| `langchain-openai` | 0.1.5 | **0.1.25** | 0.1.5 requires `core<0.2` |
| `langchain-together` | 0.1.2 | **0.1.5** | same constraint |

Because upstream pinned none of these, this is dependency *resolution*, not a departure from
a pinned configuration.

**Effect on the extraction protocol.** The dependency update does not intentionally change the
extraction protocol. Every parameter of every OpenIE/NER call is set explicitly at upstream's
own call sites rather than by library defaults:

```
named_entity_extraction_parallel.py:48  client.invoke(..., temperature=0, max_tokens=300, stop=['\n\n'], response_format={"type": "json_object"})
openie_with_retrieval_option_parallel.py:39  client.invoke(..., temperature=0, response_format={"type": "json_object"})
openie_with_retrieval_option_parallel.py:72  client.invoke(..., temperature=0, max_tokens=4096, response_format={"type": "json_object"})
langchain_util.py:30  ChatOpenAI(..., model=model_name, temperature=0.0, max_retries=5, timeout=60)
```

The model (`gpt-3.5-turbo-1106`), the prompt templates
(`src/openie_extraction_instructions.py`), the temperature (0), the token limits, the stop
sequence and the requested JSON output format are unchanged, and the exact client-library
versions are recorded in the table above and in
`scripts/hipporag/requirements-macos-arm64.txt`. Only the HTTP client library version
differs; no intentional change to the extraction protocol. Verified after the upgrade: `ChatOllama`/`ChatLlamaCpp`,
`src.named_entity_extraction_parallel`, `src.openie_with_retrieval_option_parallel`,
`src.colbertv2_indexing` and `src.hipporag.HippoRAG` all import, and
`init_langchain_model('openai','gpt-3.5-turbo-1106')` builds a client with
`temperature=0.0, max_retries=5`. `src/create_graph.py`, `src/colbertv2_knn.py`,
`src/openie_with_retrieval_option_parallel.py`, `src/named_entity_extraction_parallel.py` and
`src/ircot_hipporag.py` compile and are invoked as subprocesses exactly as upstream's shell
scripts invoke them (they use bare intra-`src` imports such as `from processing import *`,
which only resolve when run as scripts).

`pip check` reports three residual notes — `wheel`/`packaging`, and metadata claims that
`ninja` and `torch 1.13.1` are "not supported on this platform". The latter two are packaging
metadata artifacts only: both are installed and functioning, as demonstrated by the
successful 120.7s ColBERTv2 index build and the JIT-compiled C++ extensions.
