# Resource cost — size `small`  (COMPLETED jobs only)

CPU/GPU-hours = sum(elapsed x allocated units); wall_hours = stage span;
cpu_util_pct = TotalCPU / allocated core-seconds (low = over-reserved cores).
ExPert is trained outside this pipeline, so it has no cost row.

| method                 | stage      |   n_jobs |   cpu_hours |   gpu_hours |   wall_hours |   cpu_util_pct |   max_ram_GB |
|:-----------------------|:-----------|---------:|------------:|------------:|-------------:|---------------:|-------------:|
| ExPert                 | eval       |        5 |       16.5  |        0    |         0.27 |           15.4 |         20.3 |
| baselines              | eval       |       24 |        4.8  |        0    |        54.52 |           12.1 |          1.7 |
| cellot                 | eval       |       50 |     1133.59 |        0    |      1327.01 |           74.7 |         17.4 |
| cellot                 | train      |     3658 |    19470.4  |        0    |      1344.29 |           80.6 |         12   |
| fingerprint (ensemble) | eval       |     5950 |    18966.4  |        0    |      1397.14 |           87.3 |         67.6 |
| fingerprint (ensemble) | train      |     4978 |      869.21 |        0    |      1378.05 |           19.6 |         40.2 |
| fingerprint (merged)   | eval       |      432 |     1927.03 |        0    |        59.1  |           92.9 |         79.8 |
| gears                  | eval       |       26 |        8.66 |        0    |      1321.95 |           13.2 |         37.8 |
| gears                  | train      |      326 |    13138.6  |        7.35 |      1288.5  |           83   |         62.3 |
| genepert               | train+eval |       30 |       16.21 |        0    |      1223.62 |           21.5 |         39   |
| linreg (SGD)           | train+eval |       12 |       40.78 |        0    |         0.35 |           87.5 |         23   |
| scgpt                  | eval       |        1 |       15.56 |        1.94 |         1.94 |           12.4 |          1.7 |
| scgpt                  | train      |       12 |      104.09 |       13.01 |         2.33 |           14.8 |          2.1 |
| scouter                | eval       |       38 |        8.87 |        0    |      1232.71 |           14   |         40.1 |
| scouter                | train      |      477 |      434.36 |        1.61 |      1232.66 |           44.5 |         16.5 |
| XGBoost                | train+eval |       12 |      511.97 |        0    |         2.62 |           94.8 |         42.9 |
| TOTAL                  |            |    16031 |    56667    |       23.9  |       nan    |          nan   |        nan   |
