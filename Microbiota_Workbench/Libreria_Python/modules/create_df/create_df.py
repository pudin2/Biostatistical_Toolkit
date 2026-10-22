import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from ..load import load_multiple_dataframes


def get_dataframe_by_name(dfs, expected_name):
    """
    Permite encontrar el dataframe aunque el archivo tenga nombres como:
    food_groups_u24h(1), nutrients_data(1), anthro_data(1), etc.
    """

    for key in dfs.keys():
        clean_key = key.replace("(1)", "").strip()

        if clean_key == expected_name:
            return dfs[key]

    raise KeyError(
        f"No se encontró el dataframe '{expected_name}'. "
        f"Dataframes disponibles: {list(dfs.keys())}"
    )


def create_diet_dbscan_dataset(dfs):
    food = get_dataframe_by_name(dfs, "food_groups_u24h").copy()
    nutr = get_dataframe_by_name(dfs, "nutrients_data").copy()
    anthro = get_dataframe_by_name(dfs, "anthro_data").copy()

    diet_dbscan = food.merge(
        nutr[[
            "ID", "Calories", "Proteins", "Total_fat", "SFA", "PUFA",
            "Carbohydrates", "Fiber", "Na", "K"
        ]],
        on="ID",
        how="inner"
    )

    diet_dbscan = diet_dbscan.merge(
        anthro[[
            "ID", "city", "sex", "age", "bmi", "bmi_class",
            "waist", "glucose", "HDL", "triglycerides",
            "systolic_bp", "diastolic_bp", "stool_consistency",
            "medicament"
        ]],
        on="ID",
        how="left"
    )

    diet_dbscan["fiber_density"] = np.where(
        diet_dbscan["Calories"] > 0,
        diet_dbscan["Fiber"] / diet_dbscan["Calories"] * 1000,
        np.nan
    )

    diet_dbscan["Na_K_ratio"] = np.where(
        diet_dbscan["K"] > 0,
        diet_dbscan["Na"] / diet_dbscan["K"],
        np.nan
    )

    diet_dbscan["plant_foods_g"] = (
        diet_dbscan["Beans g"].fillna(0) +
        diet_dbscan["Nuts g"].fillna(0) +
        diet_dbscan["Fruits g"].fillna(0) +
        diet_dbscan["Vegetables g"].fillna(0) +
        diet_dbscan["Tubers g"].fillna(0)
    )

    diet_dbscan["animal_foods_g"] = (
        diet_dbscan["Meats g"].fillna(0) +
        diet_dbscan["Eggs g"].fillna(0) +
        diet_dbscan["Dairy g"].fillna(0)
    )

    return diet_dbscan


def save_dataframe_to_csv(df, default_filename="diet_dbscan_resultado.csv"):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    output_path = filedialog.asksaveasfilename(
        title="Guardar resultado como CSV",
        defaultextension=".csv",
        initialfile=default_filename,
        filetypes=[
            ("Archivo CSV", "*.csv"),
            ("Todos los archivos", "*.*")
        ]
    )

    if not output_path:
        output_path = default_filename

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Archivo CSV generado correctamente:")
    print(output_path)

    return output_path


# ==============================
# EJECUCIÓN PRINCIPAL
# ==============================

dfs = load_multiple_dataframes()

if dfs:
    diet_dbscan = create_diet_dbscan_dataset(dfs)

    dfs["diet_dbscan"] = diet_dbscan

    print("\nDataset diet_dbscan creado correctamente.")
    print(f"Shape final: {diet_dbscan.shape}")
    print("\nPrimeras filas:")
    print(diet_dbscan.head())

    ruta_csv = save_dataframe_to_csv(diet_dbscan)

else:
    print("No se pudo crear el dataset porque no se cargaron archivos.")
