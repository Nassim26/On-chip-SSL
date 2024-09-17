| Type                         | After 100 Train | After 500 Train | Test |
|------------------------------|-----------------|-----------------|------|
| Bias only                    | 0.27            | 0.33            | 0.42 |
| BatchNorm                    | 0.43            | 0.51            | 0.57 |
| LayerNorm                    | 0.38            | 0.49            | 0.59 |
| LayerNorm + bias             | 0.37            | 0.48            | 0.56 |
| InstanceNorm                 | 0.1             | -               | -    |
| InstanceNorm + affine        | 0.1             | -               | -    |
| InstanceNorm + affine + bias | 0.1             | -               | -    |
