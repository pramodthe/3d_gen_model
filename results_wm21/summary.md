| predictor | n | parsed | score | valid acc | failure F1 | change acc | parts exact | volume-change err | volume err | bbox err (mm) | materials J |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy | 220 | 100% | 0.672 | 73% | 0.00 | 31% | 95% | 0.485 | 0.141 | 27 | 0.86 |
| llm | 60 | 100% | 0.848 | 93% | 0.89 | 90% | 88% | 0.381 | 0.122 | 1 | 0.96 |
| oracle | 220 | 100% | 1.000 | 100% | 1.00 | 100% | 100% | 0.000 | 0.000 | 0 | 1.00 |

Score by step kind:
| predictor | normal | scaled | no_effect | wipe_body | missing_part | bad_tool |
|---|---|---|---|---|---|---|
| copy | 0.720 (n=120) | 0.655 (n=20) | 1.000 (n=20) | 0.220 (n=20) | 0.600 (n=20) | 0.600 (n=20) |
| llm | 0.876 (n=34) | 0.996 (n=6) | 1.000 (n=1) | 0.500 (n=7) | 1.000 (n=8) | 0.658 (n=4) |
| oracle | 1.000 (n=120) | 1.000 (n=20) | 1.000 (n=20) | 1.000 (n=20) | 1.000 (n=20) | 1.000 (n=20) |
