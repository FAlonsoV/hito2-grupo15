# Calcula una sola vez las probabilidades del mejor modelo (TabPFN) para una grilla de escenarios.
# La app publicada solo lee este archivo, así abre al instante y no necesita el token.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier

defectos = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
espesores = [40, 50, 60, 70, 80, 100, 120, 150, 200, 250, 300]
cintas = [1227, 1300, 1360, 1400, 1450, 1500, 1550, 1600, 1690, 1794]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["TABPFN_TOKEN"] = open("../token_tabpfn.txt").read().strip()
df = pd.read_csv("../steel_plates.csv")
df = df[df[defectos].sum(axis=1) == 1].drop(columns=["id"])            # mismo filtro del notebook
X = df.drop(columns=defectos)
y = df[defectos].values.argmax(axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2026, stratify=y)  # misma partición
modelo = Pipeline(steps=[
    ("preprocesado", ColumnTransformer([("numericas", SimpleImputer(strategy="median"), list(X.columns))])),
    ("modelo", TabPFNClassifier(random_state=2026, device="auto", fit_mode="fit_with_cache"))])
modelo.fit(X_train, y_train)

ids = list(X_test.index[:60])
if 11873 not in ids:
    ids[-1] = 11873                                                   # defecto de la demo
base = X_test.loc[ids]

# 1) Probabilidades de cada defecto con sus valores reales
orig = pd.DataFrame((modelo.predict_proba(base) * 100).round(1), columns=defectos, index=ids)
orig["espesor"] = base["Steel_Plate_Thickness"].values
orig["cinta"] = base["Length_of_Conveyer"].values
orig["acero"] = ["A300" if a == 1 else "A400" for a in base["TypeOfSteel_A300"]]
orig.index.name = "id"
orig.to_csv("originales.csv")

# 2) Probabilidades para cada combinación de espesor, cinta y acero
filas = []
for i in ids:
    for e in espesores:
        for c in cintas:
            for a in ["A300", "A400"]:
                f = X_test.loc[i].copy()
                f["Steel_Plate_Thickness"], f["Length_of_Conveyer"] = e, c
                f["TypeOfSteel_A300"] = 1 if a == "A300" else 0
                f["TypeOfSteel_A400"] = 1 - f["TypeOfSteel_A300"]
                filas.append(f)
grilla = pd.DataFrame(filas)
prob = modelo.predict_proba(grilla)
salida = pd.DataFrame((prob * 100).round(1), columns=defectos)
salida.insert(0, "id", [i for i in ids for _ in range(len(espesores) * len(cintas) * 2)])
salida.insert(1, "espesor", grilla["Steel_Plate_Thickness"].values)
salida.insert(2, "cinta", grilla["Length_of_Conveyer"].values)
salida.insert(3, "acero", ["A300" if a == 1 else "A400" for a in grilla["TypeOfSteel_A300"]])
salida.to_csv("predicciones.csv", index=False)
print("Listo:", len(salida), "escenarios")
