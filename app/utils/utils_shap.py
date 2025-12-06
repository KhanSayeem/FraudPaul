import os
from io import BytesIO
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace


def _import_shap():
    # Reduce coverage/numba conflicts by disabling coverage hooks for numba.
    os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
    # Monkey-patch coverage.types.Tracer when missing (avoids numba import crash).
    try:
        import coverage  # type: ignore
        if not hasattr(coverage, "types") or not hasattr(coverage.types, "Tracer"):
            class _DummyCoverageTypes:
                Tracer = object
                TTraceData = object

                def __getattr__(self, _name):
                    return object

            coverage.types = _DummyCoverageTypes()
    except Exception:
        pass
    # NumPy 2.x compatibility for older SHAP code paths.
    if not hasattr(np, "obj2sctype"):
        np.obj2sctype = lambda obj, default=None: np.dtype(obj).type  # type: ignore
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - env specific
        raise RuntimeError(
            f"SHAP import failed: {exc}. Try reinstalling shap or disabling coverage."
        ) from exc
    return shap


def _normalize_shap_values(
    model,
    row: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return 1D SHAP values and expected value for a single row."""
    shap = _import_shap()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        expected_value = expected_value[1]

    shap_vector = np.array(shap_values).reshape(-1)
    return shap_vector, float(expected_value)


def force_plot_html(shap_values: np.ndarray, expected_value: float, row: np.ndarray, feature_names: List[str]) -> str:
    """Return an embeddable SHAP force plot as HTML."""
    shap = _import_shap()
    force = shap.force_plot(expected_value, shap_values, row, feature_names=feature_names, matplotlib=False)
    return shap.getjs() + force.html()


def bar_plot_png(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> bytes:
    """Render a simple horizontal bar chart of top SHAP magnitudes."""
    vals = shap_values.flatten()
    order = np.argsort(np.abs(vals))[::-1][:top_n]
    ordered_features = [feature_names[i] for i in order]
    ordered_vals = vals[order]

    colors = ["#16A34A" if v >= 0 else "#DC2626" for v in ordered_vals]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(ordered_features, ordered_vals, color=colors)
    ax.set_xlabel("SHAP value impact")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def explain_row(model, row: np.ndarray, feature_names: List[str]) -> Tuple[str, bytes]:
    shap_values, expected_value = _normalize_shap_values(model, row)
    html = force_plot_html(shap_values, expected_value, row, feature_names)
    img = bar_plot_png(shap_values, feature_names)
    return html, img
