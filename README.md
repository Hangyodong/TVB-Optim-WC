# TVB-Optim-WC

42-region Wilson-Cowan E/I tuning optimization pipeline.

## Pipeline
1. Part 1 FIC: per-node c_ei tuning
2. Part 2 EIB: wLRE, wFFI optimization
3. Part 3 Gradient: full-matrix gradient (3-term loss)
4. Part 3B LowRank: low-rank delta optimization
5. Part 4 DBS: biphasic stimulation + PSD/beta analysis

## Requirements
jax, jaxlib, equinox, optax, numpy, scipy, pandas, matplotlib, tvboptim
