## locomo_amem_s1 (amem, qwen2.5:3b, 2 conversations)

| Category | n | F1 | BLEU-1 |
|---|---|---|---|
| Multi-hop | 48 | 17.67 | 10.19 |
| Temporal | 59 | 31.40 | 22.82 |
| Open-domain | 13 | 25.21 | 22.24 |
| Single-hop | 117 | 25.27 | 20.77 |
| Adversarial | 64 | 41.30 | 37.56 |
| Overall | 301 | 28.67 | 23.12 |

| Cost | Value |
|---|---|
| Memory writes | 878 |
| LLM calls per write | 2.76 |
| Seconds per write | 3.77 |
| Writes that triggered evolution | 75.6% |
| Prompt tokens per question | 1176 |
