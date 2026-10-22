"""Odds ratio (OR) con IC 95 % y prueba exacta de Fisher para variables numéricas vs un grupo binario.

Para cada variable (p. ej. un OTU) se definen dos eventos y se arma una tabla 2x2 contra los
dos grupos de `group_col`:
  - presencia: valor >= `presence_threshold`;
  - valor alto: valor >= percentil `high_quantile` de sus valores positivos.

Nada depende de nombres concretos: tabla, columna de grupo, etiquetas y variables son parámetros.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

_POSITIVE_TOKENS = {"positive", "positivo", "pos", "yes", "si", "sí", "true", "1", "case", "caso"}


def detect_blood_column(df):
    """Compatibilidad: detecta 'hidden_blood' o la errata 'hiden_blood'."""
    for col in ("hidden_blood", "hiden_blood"):
        if col in df.columns:
            return col

    raise KeyError(
        "No se encontró la columna de sangre oculta. "
        "Debe llamarse 'hidden_blood' o 'hiden_blood'. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def calculate_odds_ratio_ci(a, b, c, d):
    """
    Tabla 2x2 usada:

                      Evento sí     Evento no
    Positive             a             b
    Negative             c             d

    OR = (a/b) / (c/d) = (a*d)/(b*c)

    Si alguna celda es cero, se aplica corrección de Haldane-Anscombe:
    sumar 0.5 a todas las celdas para calcular OR e IC.
    """

    cells = np.array([a, b, c, d], dtype=float)

    correction_applied = False

    if np.any(cells == 0):
        cells = cells + 0.5
        correction_applied = True

    a_c, b_c, c_c, d_c = cells

    odds_ratio = (a_c * d_c) / (b_c * c_c)

    log_or = np.log(odds_ratio)

    se_log_or = np.sqrt(
        (1 / a_c) +
        (1 / b_c) +
        (1 / c_c) +
        (1 / d_c)
    )

    ci_lower = np.exp(log_or - 1.96 * se_log_or)
    ci_upper = np.exp(log_or + 1.96 * se_log_or)

    return odds_ratio, ci_lower, ci_upper, correction_applied


def interpret_or(odds_ratio, ci_lower, ci_upper, feature_label="variable", group_label="grupo", positive_label="Positive"):
    if pd.isna(odds_ratio):
        return "No interpretable"

    if ci_lower <= 1 <= ci_upper:
        direction = "El intervalo de confianza incluye 1; la asociación no es concluyente."
    else:
        direction = "El intervalo de confianza no incluye 1; la asociación es más consistente."

    where = f"en {group_label} = {positive_label}"

    if odds_ratio > 1:
        return f"OR > 1: el evento de {feature_label} es más frecuente {where}. " + direction
    elif odds_ratio < 1:
        return f"OR < 1: el evento de {feature_label} es menos frecuente {where}. " + direction
    else:
        return "OR = 1: no hay diferencia aparente en la frecuencia del evento. " + direction


def _resolve_labels(series, positive_label, negative_label):
    """Devuelve (positivo, negativo). Si faltan, los deduce cuando el grupo tiene exactamente 2 valores."""
    if positive_label is not None and negative_label is not None:
        return str(positive_label), str(negative_label)

    values = sorted(series.dropna().astype(str).str.strip().unique())

    if positive_label is not None:
        others = [v for v in values if v != str(positive_label)]
        if len(others) == 1:
            return str(positive_label), others[0]
    elif negative_label is not None:
        others = [v for v in values if v != str(negative_label)]
        if len(others) == 1:
            return others[0], str(negative_label)
    elif len(values) == 2:
        marked = [v for v in values if v.lower() in _POSITIVE_TOKENS]
        if len(marked) == 1:
            return marked[0], next(v for v in values if v != marked[0])

    raise ValueError(
        "No se pudieron deducir las etiquetas de los grupos. "
        f"Indica positive_label y negative_label. Valores encontrados: {values}"
    )


def odds_ratio_tables(
    df,
    group_col,
    feature_cols=None,
    id_col=None,
    positive_label=None,
    negative_label=None,
    presence_threshold=1,
    high_quantile=0.75,
    feature_label="variable",
    group_label=None,
):
    """Calcula las tablas de conteos y de OR sobre un solo DataFrame.

    feature_cols: columnas a analizar; None usa todas las numéricas (menos id y grupo).
    Devuelve (working_df, count_table, summary_table).
    """
    group_label = group_label or group_col

    if group_col not in df.columns:
        raise KeyError(f"No existe la columna de grupo '{group_col}'. Columnas disponibles: {list(df.columns)}")

    positive_label, negative_label = _resolve_labels(df[group_col], positive_label, negative_label)

    if feature_cols is None:
        excluded = {group_col, id_col}
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    else:
        feature_cols = list(feature_cols)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Estas columnas no existen: {missing}")

    if not feature_cols:
        raise ValueError("No hay columnas numéricas para analizar.")

    keep_cols = ([id_col] if id_col and id_col in df.columns else []) + [group_col] + feature_cols
    group = df[group_col].astype(str).str.strip()
    working_df = df.loc[group.isin([positive_label, negative_label]), keep_cols].copy()
    working_df[group_col] = group[working_df.index]

    pos_name = f"{group_label} = {positive_label}"
    neg_name = f"{group_label} = {negative_label}"
    is_pos = working_df[group_col] == positive_label

    count_rows = []
    summary_rows = []

    for col in feature_cols:
        values = pd.to_numeric(working_df[col], errors="coerce").fillna(0)
        working_df[col] = values

        positive_values = values[values > 0]
        high_threshold = positive_values.quantile(high_quantile) if len(positive_values) > 0 else np.nan

        criteria = [
            {
                "analysis_type": f"Presencia / ausencia de {feature_label}",
                "event_definition": f"{feature_label} PRESENTE: valor mayor o igual a {presence_threshold}",
                "no_event_definition": f"{feature_label} AUSENTE: valor menor a {presence_threshold}",
                "threshold_used": presence_threshold,
                "event_series": values >= presence_threshold,
            },
            {
                "analysis_type": f"Valor alto / no alto de {feature_label}",
                "event_definition": (
                    f"{feature_label} ALTO: valor mayor o igual al percentil {high_quantile * 100:g} de sus valores positivos"
                ),
                "no_event_definition": (
                    f"{feature_label} NO ALTO: valor por debajo del percentil {high_quantile * 100:g} de sus valores positivos"
                ),
                "threshold_used": high_threshold,
                "event_series": (
                    values >= high_threshold if not np.isnan(high_threshold) else pd.Series(False, index=values.index)
                ),
            },
        ]

        for item in criteria:
            event = item["event_series"]

            a = int((is_pos & event).sum())
            b = int((is_pos & ~event).sum())
            c = int((~is_pos & event).sum())
            d = int((~is_pos & ~event).sum())

            positive_total = a + b
            negative_total = c + d

            positive_event_pct = (a / positive_total * 100) if positive_total > 0 else np.nan
            negative_event_pct = (c / negative_total * 100) if negative_total > 0 else np.nan

            odds_ratio, ci_lower, ci_upper, correction_applied = calculate_odds_ratio_ci(a, b, c, d)
            _, fisher_p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")

            base_row = {
                f"{feature_label} analizado": col,
                "Tipo de análisis": item["analysis_type"],
                "Umbral usado para definir el evento": item["threshold_used"],
            }

            for group_name, with_event, without_event, total, pct in (
                (pos_name, a, b, positive_total, positive_event_pct),
                (neg_name, c, d, negative_total, negative_event_pct),
            ):
                count_rows.append({
                    **base_row,
                    "Grupo": group_name,
                    "Definición del evento": item["event_definition"],
                    "Número de casos CON evento": with_event,
                    "Definición del no evento": item["no_event_definition"],
                    "Número de casos SIN evento": without_event,
                    "Total de casos en este grupo": total,
                    "Porcentaje CON evento en este grupo": pct,
                })

            summary_rows.append({
                **base_row,
                "Qué significa el evento": item["event_definition"],
                f"a: {pos_name} y CON evento": a,
                f"b: {pos_name} y SIN evento": b,
                f"c: {neg_name} y CON evento": c,
                f"d: {neg_name} y SIN evento": d,
                f"Total {pos_name}": positive_total,
                f"Total {neg_name}": negative_total,
                f"% con evento en {pos_name}": positive_event_pct,
                f"% con evento en {neg_name}": negative_event_pct,
                "Odds Ratio OR": odds_ratio,
                "Límite inferior IC 95% del OR": ci_lower,
                "Límite superior IC 95% del OR": ci_upper,
                "p-valor prueba exacta de Fisher": fisher_p_value,
                "¿Se aplicó corrección 0.5 por celdas en cero?": "Sí" if correction_applied else "No",
                "Interpretación del OR": interpret_or(
                    odds_ratio, ci_lower, ci_upper, feature_label, group_label, positive_label
                ),
            })

    return working_df, pd.DataFrame(count_rows), pd.DataFrame(summary_rows)


def odds_ratio_from_loaded(
    dfs,
    group_df_name,
    group_col,
    value_df_name=None,
    feature_cols=None,
    id_col_group="ID",
    id_col_value="ID",
    positive_label=None,
    negative_label=None,
    presence_threshold=1,
    high_quantile=0.75,
    feature_label="variable",
    group_label=None,
    verbose=True,
):
    """Igual que los demás análisis `*_from_loaded`: trabaja sobre el diccionario `dfs`.

    group_df_name / group_col: tabla y columna con los dos grupos a comparar.
    value_df_name: tabla con las variables (None = la misma tabla de los grupos); se une por ID.
    Devuelve (working_df, count_table, summary_table).
    """
    for name in (group_df_name, value_df_name or group_df_name):
        if name not in dfs:
            raise KeyError(f"No existe '{name}' en dfs. Disponibles: {list(dfs.keys())}")

    group_df = dfs[group_df_name]

    if value_df_name is None or value_df_name == group_df_name:
        data, id_col = group_df, id_col_group if id_col_group in group_df.columns else None
    else:
        value_df = dfs[value_df_name]
        for col, frame, name in ((id_col_group, group_df, group_df_name), (id_col_value, value_df, value_df_name)):
            if col not in frame.columns:
                raise KeyError(f"La columna '{col}' no existe en '{name}'")

        left = group_df[[id_col_group, group_col]].copy()
        right = value_df.copy()
        left[id_col_group] = left[id_col_group].astype(str).str.strip()
        right[id_col_value] = right[id_col_value].astype(str).str.strip()
        data = left.merge(right, left_on=id_col_group, right_on=id_col_value, how="inner", suffixes=("", "_valores"))
        id_col = id_col_group

    working_df, count_table, summary_table = odds_ratio_tables(
        data,
        group_col=group_col,
        feature_cols=feature_cols,
        id_col=id_col,
        positive_label=positive_label,
        negative_label=negative_label,
        presence_threshold=presence_threshold,
        high_quantile=high_quantile,
        feature_label=feature_label,
        group_label=group_label,
    )

    if verbose:
        print(f"OR calculado para {summary_table.iloc[:, 0].nunique()} variables, {len(working_df)} casos.")

    return working_df, count_table, summary_table


def build_otu_2x2_tables_from_complete_df(
    df,
    otu_cols=("Otu00043", "Otu00036"),
    id_col="ID",
    blood_col=None,
    positive_label="Positive",
    negative_label="Negative",
):
    """Compatibilidad con el uso original (OTUs vs sangre oculta)."""
    return odds_ratio_tables(
        df,
        group_col=blood_col or detect_blood_column(df),
        feature_cols=otu_cols,
        id_col=id_col,
        positive_label=positive_label,
        negative_label=negative_label,
        feature_label="OTU",
        group_label="sangre oculta",
    )


def save_tables_to_csv(count_table, summary_table, output_dir=None, prefix="odds_ratio"):
    """Guarda las dos tablas como CSV. Sin `output_dir` se pide con un diálogo (tkinter)."""
    if output_dir is None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            output_dir = filedialog.askdirectory(title="Selecciona la carpeta donde guardar las tablas CSV")
        finally:
            root.destroy()

    output_dir = Path(output_dir or ".")

    count_path = output_dir / f"{prefix}_conteos.csv"
    summary_path = output_dir / f"{prefix}_or_ic95.csv"

    count_table.to_csv(count_path, index=False, encoding="utf-8-sig")
    summary_table.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return count_path, summary_path
