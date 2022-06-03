"""

"""
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error


def get_residuals(
        y_pred: List[float],
        y_true: List[float],
        _norm: int = 2,
    ) -> Tuple[
        float,
        List[float],
        List[float],
        List[float],
    ]:
    """ Routine to calculate the residuals. 
        Mby. the norm is seriously included in futur.

    Args:
        y_pred (List[float]): Predicted values.
        y_true (List[float]): True values (target).
        _norm (int, optional): Defaults to 2 for MSE.

    Returns:
        Tuple[ float, List[float], List[float], List[float], ]
    """
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    # Sanity checks
    assert y_pred.shape[0] > 0
    assert y_true.shape[0] > 0
    assert len(y_pred) == len(y_pred)
    
    _mse = mean_squared_error(y_pred, y_true)
    resid = np.sort((y_pred - y_true) ** _norm)
    cumsum = resid.cumsum()
    cumsum_perc = cumsum / resid.sum()

    return _mse, resid, cumsum, cumsum_perc


def visualize_residuals(
        df_preds: pd.DataFrame, 
        y_true: List, 
        log_yaxes: bool = False,
    ) -> Tuple[
        go.Figure,
        List[float],
        List[float],
    ]:
    """Generates the main figure. 
        Currently, it need the true values to calculate MSE.

    Args:
        df_preds (pd.DataFrame): Predictedvalues of each model.
        y_true (List): True Values.
        log_yaxes (bool, optional): Defaults to False.

    Returns:
        Tuple[ go.Figure, List[float], List[float], ]: Figure, residuals, cumsum.
    """
    assert len(df_preds.columns) > 0
    assert len(y_true) > 0

    # Create figure
    _fig = go.Figure()

    mses, resids, cumsum_percs, columns = [], [], [], []
    for column in df_preds.columns:
        mse, resid, _, cumsum_perc = get_residuals(df_preds[column], y_true)
        mses.append(mse)
        resids.append(resid)
        cumsum_percs.append(cumsum_perc)
        columns.append(column)

        _fig = _fig.add_trace(go.Scatter(
            y=resid,
            x=np.linspace(0, 100, len(resid)),
            name=f"{column} (mse={mse.round(2)})",
            mode="markers+lines",
            marker=dict(size=3,)
        ))

    _fig = _fig.update_layout(
        xaxis=dict(
            title=r"% of data",
            tickmode='linear',
            tick0=0.0,
            dtick=10,
            rangemode="tozero",
            range=[0, 100],
        ),
        yaxis=dict(
            title=r"$(\hat{y}-y)^{2}$",
            rangemode="tozero",
        ),
        showlegend=True,
        legend=dict(
            title="Vanilla Model",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.01,
        ),
        template="simple_white",
    )

    if log_yaxes == True:
        _fig = _fig.update_yaxes(type="log")

    return _fig, resids, cumsum_percs
