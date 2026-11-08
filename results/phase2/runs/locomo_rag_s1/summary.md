## locomo_rag_s1 (rag, qwen2.5:3b, 2 conversations)

| Category | n | F1 | BLEU-1 |
|---|---|---|---|
| Multi-hop | 48 | 16.64 | 12.18 |
| Temporal | 59 | 30.47 | 24.82 |
| Open-domain | 13 | 18.19 | 17.76 |
| Single-hop | 117 | 22.39 | 18.22 |
| Adversarial | 64 | 36.47 | 34.08 |
| Overall | 301 | 25.87 | 21.90 |

| Cost | Value |
|---|---|
| Memory writes | 878 |
| LLM calls per write | 0.00 |
| Seconds per write | 0.01 |
| Writes that triggered evolution | 0.0% |
| Prompt tokens per question | 703 |
