## CPSC 424 Final Project

Parallel N-Body Algorithms


### Results

See results at [`results.ipynb`](https://github.com/areebg9/cpsc424-final/blob/main/results.ipynb).

Example output for `./nbody 100000`:

```
Calculate gravitational forces between 100000 random bodies:
Brute force O(n^2) sequential approach:
Time taken: 17.5644 s
Body #33333 force: (2.1273e-05, -1.33967e-05)
Body #66666 force: (9.45141e-07, -7.18416e-06)
Body #99999 force: (5.15165e-06, -4.28195e-06)

Brute force OpenMP parallel approach (memory-intensive):
Using 20 threads...
Time taken: 2.3214 s
Body #33333 force: (2.1273e-05, -1.33967e-05)
Body #66666 force: (9.45141e-07, -7.18416e-06)
Body #99999 force: (5.15165e-06, -4.28195e-06)

Brute force OpenMP parallel approach (memory-efficient):
Using 20 threads...
Time taken: 3.54985 s
Body #33333 force: (2.1273e-05, -1.33967e-05)
Body #66666 force: (9.45141e-07, -7.18416e-06)
Body #99999 force: (5.15165e-06, -4.28195e-06)

Brute force ParlayLib parallel approach (memory-intensive):
Using 20 workers...
Time taken: 1.90592 s
Body #33333 force: (2.1273e-05, -1.33967e-05)
Body #66666 force: (9.45141e-07, -7.18416e-06)
Body #99999 force: (5.15165e-06, -4.28195e-06)

Brute force ParlayLib parallel approach (memory-efficient):
Using 20 workers...
Time taken: 3.61583 s
Body #33333 force: (2.1273e-05, -1.33967e-05)
Body #66666 force: (9.45141e-07, -7.18416e-06)
Body #99999 force: (5.15165e-06, -4.28195e-06)
```

![result-1](https://lh3.googleusercontent.com/pw/AP1GczOF9_-GVf_ZvR9bR8SqDG1Nmt26x-xalRekrlKm0Xvv2ANn2rWZ9TzenX3XH5LFmgWbS0Wpc6Dr-nUIOkIZM_j-Nrwe9ohWcjdShxOr-3KyHpIzcjirBTNVQaO2ZGxXHwAQ89cKiG4wNd6nXclOzmdi=w720-h432-s-no-gm?authuser=0)
