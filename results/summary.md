# Results summary (generated)

## False alarms on held-out healthy prompts (target ARL₀ = 10,000)

| detector        |   test_arl0 |
|:----------------|------------:|
| cusum           |     4167.93 |
| e_per_tenant    |      inf    |
| e_pooled        |      inf    |
| fixed_threshold |     4456.73 |
| http_breaker    |      inf    |

## Severity 0.1: median delay (requests) / detection rate

| fault              | cusum      | e_per_tenant   | e_pooled   | fixed_threshold   |
|:-------------------|:-----------|:---------------|:-----------|:------------------|
| output_cap         | 473 / 90%  | 1639 / 70%     | 708 / 12%  | 259 / 85%         |
| quant_swap         | 84 / 88%   | 674 / 97%      | 519 / 100% | 499 / 85%         |
| truncate_context   | 780 / 55%  | 969 / 38%      | 934 / 3%   | 1020 / 53%        |
| drop_system        | 753 / 60%  | 1337 / 50%     | 1321 / 5%  | 747 / 75%         |
| sampling           | 746 / 85%  | — / 0%         | 1208 / 2%  | 972 / 47%         |
| model_substitution | 1337 / 52% | 256 / 3%       | — / 0%     | 1355 / 47%        |
| throttle           | 979 / 45%  | — / 0%         | — / 0%     | 1083 / 37%        |

## Severity 0.3: median delay (requests) / detection rate

| fault              | cusum      | e_per_tenant   | e_pooled   | fixed_threshold   |
|:-------------------|:-----------|:---------------|:-----------|:------------------|
| output_cap         | 48 / 88%   | 123 / 100%     | 315 / 100% | 73 / 80%          |
| quant_swap         | 24 / 85%   | 89 / 100%      | 45 / 100%  | 72 / 90%          |
| truncate_context   | 377 / 88%  | 163 / 100%     | 1248 / 68% | 202 / 90%         |
| drop_system        | 112 / 90%  | 135 / 100%     | 534 / 97%  | 83 / 95%          |
| sampling           | 223 / 87%  | 1190 / 13%     | 2083 / 18% | 663 / 83%         |
| model_substitution | 832 / 80%  | 1063 / 82%     | 2039 / 10% | 286 / 87%         |
| throttle           | 1637 / 42% | — / 0%         | — / 0%     | 1381 / 33%        |

## Severity 1.0: median delay (requests) / detection rate

| fault              | cusum      | e_per_tenant   | e_pooled   | fixed_threshold   |
|:-------------------|:-----------|:---------------|:-----------|:------------------|
| output_cap         | 8 / 87%    | 24 / 100%      | 20 / 100%  | 19 / 83%          |
| quant_swap         | 5 / 93%    | 18 / 100%      | 6 / 100%   | 25 / 83%          |
| truncate_context   | 22 / 92%   | 28 / 100%      | 32 / 100%  | 44 / 83%          |
| drop_system        | 11 / 93%   | 27 / 100%      | 25 / 100%  | 28 / 87%          |
| sampling           | 32 / 85%   | 131 / 100%     | 210 / 100% | 73 / 88%          |
| model_substitution | 37 / 88%   | 66 / 100%      | 103 / 100% | 52 / 85%          |
| throttle           | 1360 / 45% | — / 0%         | — / 0%     | 1669 / 43%        |

## Attribution: share of runs blaming provider A

| method     |   none |   provider_fault |   traffic_shift |   both |
|:-----------|-------:|-----------------:|----------------:|-------:|
| pooled     |      0 |             0.78 |            1    |   1    |
| per_tenant |      0 |             0.89 |            1    |   1    |
| ours       |      0 |             0.78 |            0.99 |   1    |
| ours_input |      0 |             0.79 |            0    |   0.77 |

Overall correct blame decisions:

| method     |   correct |
|:-----------|----------:|
| ours       |     0.698 |
| ours_input |     0.89  |
| per_tenant |     0.723 |
| pooled     |     0.696 |

## Ablation: detection rate by signal group

| signals               |   output_cap |   quant_swap |   truncate_context |   drop_system |   sampling |   model_substitution |   throttle |
|:----------------------|-------------:|-------------:|-------------------:|--------------:|-----------:|---------------------:|-----------:|
| all incl. latency     |         0.83 |         0.95 |               0.85 |          0.87 |       0.83 |                 0.88 |       0.92 |
| latency               |         0.62 |         0.63 |               0.6  |          0.65 |       0.68 |                 0.7  |       0.63 |
| length                |         1    |         1    |               0.98 |          1    |       0.5  |                 0.68 |       0.02 |
| quality (all content) |         1    |         1    |               0.7  |          1    |       0.12 |                 0.2  |       0    |
| refusal               |         0    |         0    |               0    |          0    |       0    |                 0    |       0    |
| repetition            |         0    |         1    |               0.02 |          0.02 |       0    |                 0.02 |       0    |
| tools                 |         0    |         0.65 |               0    |          0    |       0    |                 0    |       0    |
