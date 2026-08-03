# Semantic Dependency Consolidation Design

## Purpose
Consolidate the scattered `requirements-phase*.txt` files into a single `pyproject.toml` managed by `uv`. Transition from phase-based naming to functional, semantic groupings.

## Architecture

We will use `[project.optional-dependencies]` in `pyproject.toml` to maintain modularity without restructuring the project into a monorepo workspace.

### Dependency Groups
1. **Base `dependencies`**: Core tools needed for data handling.
   - `numpy>=2.0,<3.0`
   - `opencv-python>=4.10,<5.0`
   - `PyYAML>=6.0,<7.0`
   - `tqdm>=4.66,<5.0`
2. **`classical` (Optional)**: Tools for classical ML and CV baselines.
   - `scikit-image>=0.25,<0.27`
   - `scikit-learn>=1.5,<2.0`
   - `joblib>=1.4,<2.0`
   - `matplotlib>=3.8,<4.0`
3. **`detector` (Optional)**: Deep learning frameworks for training/inference.
   - `torch>=2.2,<3.0`
   - `torchvision>=0.17,<1.0`
   - `ultralytics>=8.3,<9.0`
   - `Pillow>=10.0,<13.0`
4. **`video` (Optional)**: Temporal tracking tools.
   - Inherits `detector` (i.e. `"firevision-case-study[detector]"`)
   - `scipy>=1.11,<2.0`
5. **Dev Group**: Testing and development tools.
   - `pytest>=8.0,<9.0` (in `[dependency-groups]`)

## Changes Required

1. **`pyproject.toml`**: Rewrite to include `dependencies` and `optional-dependencies` as specified above.
2. **Clean up**: Delete `.venv` and all `requirements-*.txt` files.
3. **`.gitignore`**: Update `.gitignore` to ensure `.uv/` or other cache artifacts are ignored.
4. **`Makefile`**: Rewrite targets to use `uv run` and `uv sync` with the appropriate `--extra` flags depending on the functional group being invoked.

## Self-Review Checklist
- [x] Placeholders: None.
- [x] Consistency: Optional dependencies align with the old phases accurately.
- [x] Scope: Highly focused on dependency management with `uv`.
- [x] Ambiguity: Explicit library versions retained.
