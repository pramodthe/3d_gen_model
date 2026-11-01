| predictor | n | parsed | score | valid acc | failure F1 | change acc | parts exact | volume-change err | volume err | bbox err (mm) | materials J |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy | 220 | 100% | 0.664 | 73% | 0.00 | 29% | 96% | 0.514 | 0.143 | 27 | 0.86 |
| llm | 60 | 100% | 0.847 | 93% | 0.88 | 88% | 88% | 0.386 | 0.129 | 2 | 0.97 |
| oracle | 220 | 100% | 1.000 | 100% | 1.00 | 100% | 100% | 0.000 | 0.000 | 0 | 1.00 |

Score by step kind:
| predictor | normal | scaled | no_effect | wipe_body | missing_part | bad_tool |
|---|---|---|---|---|---|---|
| copy | 0.709 (n=120) | 0.635 (n=20) | 1.000 (n=20) | 0.220 (n=20) | 0.600 (n=20) | 0.600 (n=20) |
| llm | 0.887 (n=34) | 0.987 (n=6) | 1.000 (n=1) | 0.500 (n=7) | 1.000 (n=8) | 0.557 (n=4) |
| oracle | 1.000 (n=120) | 1.000 (n=20) | 1.000 (n=20) | 1.000 (n=20) | 1.000 (n=20) | 1.000 (n=20) |
