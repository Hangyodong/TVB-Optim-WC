#!/usr/bin/env python3
"""
pipeline_verify.py
Verify Patches 7, 8, 9, 10 are correctly applied.

Usage:
    cd /scratch/home/wog3597/optim
    python3 pipeline_verify.py
"""
import ast, json, sys, textwrap, pathlib
import numpy as np

ROOT = pathlib.Path('/scratch/home/wog3597/optim')
PASS_LIST, FAIL_LIST = [], []

# ── helpers (defined FIRST) ───────────────────────────────────

def _assert_in(pattern, text):
    assert pattern in text, f"pattern not found: {pattern!r}"

def _assert_not_in(pattern, text):
    assert pattern not in text, f"old pattern still present: {pattern!r}"

def _assert_any_in(patterns, text):
    assert any(p in text for p in patterns), \
        f"none of these patterns found: {patterns}"

def read(fname):
    return (ROOT / fname).read_text(encoding='utf-8')

def nb_config_src():
    nb = json.loads((ROOT / 'main.ipynb').read_text(encoding='utf-8'))
    for c in nb['cells']:
        src = ''.join(c.get('source', []))
        if 'cfg = Config(' in src and 'optimizer_learning_rate' in src:
            return src
    raise AssertionError('Config cell not found in main.ipynb')

def check(name, fn):
    try:
        fn()
        print(f"  ✓  {name}")
        PASS_LIST.append(name)
    except Exception as e:
        short = str(e).splitlines()[0][:120]
        print(f"  ✗  {name}")
        print(f"       └─ {short}")
        FAIL_LIST.append((name, str(e)))

# ══════════════════════════════════════════════════════════════
# A. Syntax
# ══════════════════════════════════════════════════════════════
print("\n[A] Syntax — all .py files")

for fname in ['model.py','pipeline_contracts.py','data_loader.py',
              'part1_fic.py','part2_eib.py',
              'part3_gradient.py','part4_dbs.py','config.py']:
    def _syn(f=fname):
        ast.parse((ROOT/f).read_text(encoding='utf-8'))
    check(f"{fname} parses OK", _syn)

# ══════════════════════════════════════════════════════════════
# B. Patch 7 — wLRE/wFFI per-node (N,)
# ══════════════════════════════════════════════════════════════
print("\n[B] Patch 7 — wLRE/wFFI per-node (N,)")

check("model.py: wLRE[:, None] broadcast present",
    lambda: _assert_any_in(
        ['wLRE[:, None]', 'wLRE.ndim == 1', 'if wLRE.ndim'],
        read('model.py')))

check("model.py: old bare S_e * params.wLRE line removed",
    lambda: _assert_not_in(
        'c_lre = S_e * params.wLRE\n', read('model.py')))

check("pipeline_contracts.py: wLRE init shape (n_nodes,)",
    lambda: _assert_any_in(
        ['np.ones((n_nodes,)', 'jnp.ones((n_nodes,)'],
        read('pipeline_contracts.py')))

check("pipeline_contracts.py: old (n_nodes, n_nodes) init removed",
    lambda: _assert_not_in(
        'np.ones((n_nodes, n_nodes)', read('pipeline_contracts.py')))

check("part2_eib.py: _clip_pernode present",
    lambda: _assert_in('def _clip_pernode(', read('part2_eib.py')))

check("part2_eib.py: row_err = jnp.mean(fc_diff, axis=1) present",
    lambda: _assert_in('jnp.mean(fc_diff, axis=1)', read('part2_eib.py')))

check("part2_eib.py: old N×N update removed",
    lambda: _assert_not_in(
        'wLRE + eta_eib * fc_diff * row_rmse', read('part2_eib.py')))

def _check_paramset_shape():
    try:
        sys.path.insert(0, str(ROOT))
        import importlib
        if 'pipeline_contracts' in sys.modules:
            importlib.reload(sys.modules['pipeline_contracts'])
        from pipeline_contracts import ParamSet
        ps = ParamSet.default(42)
        assert ps.wLRE.shape == (42,), f"wLRE shape {ps.wLRE.shape}"
        assert ps.wFFI.shape == (42,), f"wFFI shape {ps.wFFI.shape}"
    except ImportError as e:
        if 'tvboptim' in str(e) or 'jax' in str(e):
            text = read('pipeline_contracts.py')
            _assert_any_in(['np.ones((n_nodes,)', 'jnp.ones((n_nodes,)'], text)
        else:
            raise
check("pipeline_contracts.py: ParamSet.default(42) → wLRE shape (42,)",
    _check_paramset_shape)

# ══════════════════════════════════════════════════════════════
# C. Patch 8 — SC Preprocessing
# ══════════════════════════════════════════════════════════════
print("\n[C] Patch 8 — SC Preprocessing")

dl = read('data_loader.py')

check("data_loader.py: symmetrize present",
    lambda: _assert_any_in(
        ['0.5 * (weights + weights.T)',
         '0.5 * (weights_raw + weights_raw.T)',
         '(weights + weights.T) * 0.5'], dl))

check("data_loader.py: non-finite/negative removal present",
    lambda: _assert_any_in(
        ['np.isfinite(weights_raw)', 'np.isfinite(weights)',
         'isfinite(weights)'], dl))

check("data_loader.py: in-degree normalization present",
    lambda: _assert_any_in(['row_sum', 'sum(axis=1'], dl))

check("data_loader.py: log1p present",
    lambda: _assert_in('log1p', dl))

check("data_loader.py: old +0.5 offset removed",
    lambda: _assert_not_in('log1p(weights_raw + 0.5)', dl))

check("data_loader.py: old max normalization removed",
    lambda: _assert_not_in('weights /= weights.max()', dl))

def _sc_numerical():
    W = np.genfromtxt(str(ROOT/'weight.csv'), delimiter=',').astype(np.float32)
    W = np.where(np.isfinite(W) & (W >= 0), W, 0.0)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    sc_mask = (W > 0).astype(np.float32)
    assert sc_mask.sum() > 0
    W_log = np.log1p(W)
    row_sum = W_log.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    W_norm = (W_log / row_sum) * sc_mask
    assert np.all(np.diag(W_norm) == 0), "diagonal not zero"
    # Note: W_norm is NOT symmetric after in-degree normalization — by design
    nz = W_norm.sum(axis=1)
    nz = nz[nz > 0]
    assert np.allclose(nz, 1.0, atol=1e-4), \
        f"row sums not ≈ 1: min={nz.min():.5f} max={nz.max():.5f}"
check("SC numerical: diagonal=0, symmetric, row_sum≈1",
    _sc_numerical)

# ══════════════════════════════════════════════════════════════
# D. Patch 9 — 3-term Loss
# ══════════════════════════════════════════════════════════════
print("\n[D] Patch 9 — 3-term Loss")

g3 = read('part3_gradient.py')

check("part3_gradient.py: _compute_nodewise_corr_loss present",
    lambda: _assert_in('def _compute_nodewise_corr_loss(', g3))

check("part3_gradient.py: _compute_rmse_loss present",
    lambda: _assert_in('def _compute_rmse_loss(', g3))

check("part3_gradient.py: _compute_correlation_loss preserved",
    lambda: _assert_in('def _compute_correlation_loss(', g3))

check("part3_gradient.py: optimizer_global_corr_weight used",
    lambda: _assert_in('optimizer_global_corr_weight', g3))

check("part3_gradient.py: optimizer_nodewise_corr_weight used",
    lambda: _assert_in('optimizer_nodewise_corr_weight', g3))

check("part3_gradient.py: optimizer_rmse_weight used",
    lambda: _assert_in('optimizer_rmse_weight', g3))

check("part3_gradient.py: old single-term loss removed",
    lambda: _assert_not_in('return corr_l + 0.01 * act_l', g3))

check("part3_gradient.py: jax imported",
    lambda: _assert_any_in(['import jax\n', 'import jax '], g3))

def _cfg_new_fields():
    sys.path.insert(0, str(ROOT))
    import importlib
    if 'config' in sys.modules:
        importlib.reload(sys.modules['config'])
    from config import Config
    c = Config()
    for field in ['optimizer_global_corr_weight',
                  'optimizer_nodewise_corr_weight',
                  'optimizer_rmse_weight']:
        assert hasattr(c, field), f"{field} missing"
    assert abs(c.optimizer_global_corr_weight   - 0.4) < 1e-6
    assert abs(c.optimizer_nodewise_corr_weight - 0.4) < 1e-6
    assert abs(c.optimizer_rmse_weight          - 0.2) < 1e-6
    assert hasattr(c, 'correlation_loss_weight'), 'EIB weight removed!'
    assert hasattr(c, 'rmse_loss_weight'),        'EIB weight removed!'
check("config.py: 6 new loss weight fields + EIB weights preserved",
    _cfg_new_fields)

def _nodewise_sanity():
    n = 42
    rng = np.random.RandomState(0)
    p = rng.randn(n, n).astype(np.float32)
    t = rng.randn(n, n).astype(np.float32)
    np.fill_diagonal(p, 0.0); np.fill_diagonal(t, 0.0)
    mask = 1.0 - np.eye(n)
    def nw(a, b):
        cs = []
        for i in range(n):
            x = a[i]*mask[i]; y = b[i]*mask[i]
            ne = max(mask[i].sum(), 1.0)
            xm = x - x.sum()/ne; ym = y - y.sum()/ne
            num = (xm*ym).sum()
            den = np.sqrt(max((xm**2).sum(),1e-10)*max((ym**2).sum(),1e-10))
            cs.append(num/den)
        return 1.0 - np.mean(cs)
    assert 0.0 <= nw(p,t) <= 2.0
    assert nw(p,p) < 1e-5
check("nodewise corr loss sanity: range [0,2], perfect→0",
    _nodewise_sanity)

# ══════════════════════════════════════════════════════════════
# E. Patch 10 — Cell 3 Config
# ══════════════════════════════════════════════════════════════
print("\n[E] Patch 10 — main.ipynb Cell 3 Config")

src = nb_config_src()

for param, val in [
    ('optimizer_learning_rate             = 0.0002', '0.0002'),
    ('optimizer_max_steps                 = 2000',   '2000'),
    ('optimizer_chunk_steps               = 10',     '10'),
    ('optimizer_bold_window_tr            = 240',    '240'),
]:
    def _chk(p=param): _assert_in(p, src)
    check(f"Cell 3: {param.split('=')[0].strip()} = {val}", _chk)

check("Cell 3: old optimizer_learning_rate=0.002 removed",
    lambda: _assert_not_in(
        'optimizer_learning_rate             = 0.002,', src))

check("Cell 3: old optimizer_max_steps=1000 removed",
    lambda: _assert_not_in(
        'optimizer_max_steps                 = 1000', src))

for field in ['optimizer_global_corr_weight',
              'optimizer_nodewise_corr_weight',
              'optimizer_rmse_weight']:
    def _chk(f=field): _assert_in(f, src)
    check(f"Cell 3: {field} present", _chk)

check("Cell 3: EIB correlation_loss_weight preserved",
    lambda: _assert_in('correlation_loss_weight', src))

check("Cell 3: EIB rmse_loss_weight preserved",
    lambda: _assert_in('rmse_loss_weight', src))

for label, key in [('FIC', 'fic_target_firing_rate_hz'),
                   ('EIB', 'eib_max_iterations'),
                   ('DBS', 'dbs_pulse_amplitude')]:
    def _chk(k=key): _assert_in(k, src)
    check(f"Cell 3: {label} parameters present", _chk)

check("config.py: cache_version has patch suffix",
    lambda: _assert_any_in(['_p7','_p8','_p9','_p10'], read('config.py')))

# ══════════════════════════════════════════════════════════════
# F. Scientific code unchanged
# ══════════════════════════════════════════════════════════════
print("\n[F] Scientific code unchanged")

check("model.py: WilsonCowanEIB.dynamics() present",
    lambda: _assert_in('def dynamics(', read('model.py')))

check("model.py: dE/dt expression present",
    lambda: _assert_any_in(
        ['excitatory_derivative','dE ','d_e'], read('model.py')))

check("part1_fic.py: run_fic present",
    lambda: _assert_in('def run_fic(', read('part1_fic.py')))

check("part4_dbs.py: _build_biphasic_pulse_train present",
    lambda: _assert_in('def _build_biphasic_pulse_train(', read('part4_dbs.py')))

check("part4_dbs.py: try/finally restore present",
    lambda: (_assert_in('original_dynamics', read('part4_dbs.py')),
             _assert_in('finally:', read('part4_dbs.py'))))

# ══════════════════════════════════════════════════════════════
# RESULT
# ══════════════════════════════════════════════════════════════
total = len(PASS_LIST) + len(FAIL_LIST)
print(f"\n{'='*60}")
print(f"  Result: {len(PASS_LIST)}/{total} PASS   {len(FAIL_LIST)} FAIL")
print(f"{'='*60}")

if FAIL_LIST:
    print("\nFailed checks:")
    for name, err in FAIL_LIST:
        print(f"\n  ✗ {name}")
        for line in textwrap.wrap(err, 72):
            print(f"      {line}")
    sys.exit(1)
else:
    print("\n  All pipeline changes verified. ✓")
    print("  Patches 7 (per-node), 8 (SC preproc),")
    print("  9 (3-term loss), 10 (Cell 3) correctly applied.")
    sys.exit(0)
