# Work carried on `local` that is not in ml-explore/mlx-lm

`local` = upstream release tag v0.32.2 plus the upstream commits and `topic/*`
branches below, merged with `--no-ff`. `main` mirrors `ml-explore/mlx-lm` `main`
(fast-forward only). Each topic branch is one change on top of v0.32.2.

| Topic branch | What | Upstream | Original branch (PR head, kept) |
|---|---|---|---|
| `topic/qwen3-coder-bare-function` | recover Qwen3-Coder tool calls written without the opening `<tool_call>` marker | not submitted (eejd/py-mlx-lm#3) | `fix/qwen3-coder-bare-function` |
| `topic/mistral-tool-parser-from-vocab` | infer the mistral tool parser from the `[TOOL_CALLS]` vocab token | ml-explore/mlx-lm#1969, closed (withdrawn 2026-10-16); eejd/py-mlx-lm#1 | `fix/mistral-tool-parser-from-vocab` |
| `topic/batch-kv-cache-fast-mask` | skip the attention mask in batch caches with no padding | ml-explore/mlx-lm#1970, closed (withdrawn 2026-10-16); eejd/py-mlx-lm#2 | `fix/batch-kv-cache-fast-mask` |
| `topic/llama3-tool-parser` | tool parser for Llama 3.1+ JSON calls (`<\|python_tag\|>{...}` and bare JSON); `ToolCallFormatter` can return a parser's non-call text as content | not submitted: upstream candidate only, after the quantbench + agentbench soak (eejd/py-mlx-lm#5) | - |
| `topic/server-ignore-eos` | `ignore_eos` request option on `/v1/completions` and `/v1/chat/completions` (EOS tokens do not end the request; `max_tokens` and `stop` still do; default off), for fixed-output load tests | not submitted (eejd/py-mlx-lm#12) | - |

Not on `local` (not carried by any port): `fix/gpt-oss-harmony-channel-parsing`
(ml-explore/mlx-lm#1867, open; eejd/py-mlx-lm#4). Its branch is still the upstream PR's head.

The `fix/*` branches are the heads of the upstream PRs above and are left untouched; the
`topic/*` branches are their content rebased onto `main` (v0.32.2).

## Merge of upstream 4cfc647

The merge includes the 12 upstream commits through `4cfc647` (Flint support).
Upstream's TextMachine refactor replaced tuple-state `split`/`step` with
`TextStateMachine.reader().feed()` and `Delta`. The server now applies the
Qwen3-Coder bare-call filter to each content segment around state transitions,
retains the Llama 3 formatter's non-call content fallback, and preserves
`ignore_eos` independently for batched and single generation on the new API.
The token stop-sequence helper remains for the batch-generation API and its
tests. Mistral vocabulary detection and the batch KV-cache fast-mask path remain
carried unchanged.

The setuptools-scm `git describe` command excludes the fork's `-local*` tags:
their names are not PEP 440 versions and upstream's new version machinery cannot
parse them. Version discovery still uses the upstream version tags.

## Tags


`v0.32.2-local01` (superseded: its qwen3-coder port raised `UnboundLocalError` for any request with tools to a bare-call parser); `v0.32.2-local02` and `v0.32.2-local03` (tests only over local02; the flush now runs per segment, with handler-level tests; local03 is superseded by local04); `v0.32.2-local04` (local03 + `topic/llama3-tool-parser`; superseded by local05; **behaviour change for callers of `mlx_lm.server.ToolCallFormatter`:** it returns `(tool_calls, content)` instead of a list); `v0.32.2-local05` (local04 + `topic/server-ignore-eos`; new `ignore_eos` request option, default off); `v0.32.2-local06` (not yet cut: merge of upstream 4cfc647).
