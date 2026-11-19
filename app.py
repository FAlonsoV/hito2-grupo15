# Simulador what-if (Hito 2) - versión web: lee las probabilidades ya calculadas por TabPFN
# (precalcular.py usa el mismo modelo, filtro y partición del notebook)
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Simulador K_Scatch", page_icon="🔩", layout="wide")

defectos = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
critica = "K_Scatch"
riesgo_promedio = 18.7        # % promedio de K_Scatch en el test (bloque 15 del notebook)

# ---------- Estilo (colores de la presentación: azul acero y ámbar) ----------
st.markdown("""
<style>
[data-testid="stToolbar"], footer {visibility: hidden;}
.block-container {padding-top: 1.5rem;}
.hero {background: linear-gradient(120deg, #1F2D3A 0%, #4A6274 100%); color: white;
       padding: 1.6rem 2rem; border-radius: 14px; border-bottom: 5px solid #E8A845; margin-bottom: 1.2rem;}
.hero h1 {color: white; font-size: 2rem; margin: 0;}
.hero p {color: #D6DEE5; margin: .3rem 0 0 0;}
.tarjeta {background: white; border-radius: 12px; padding: 1rem 1.2rem; border: 1px solid #E3E8ED;
          box-shadow: 0 2px 8px rgba(31,45,58,.08); height: 100%;}
.tarjeta .titulo {color: #6B7C8C; font-size: .85rem; text-transform: uppercase; letter-spacing: .04em;}
.tarjeta .valor {color: #1F2D3A; font-size: 2rem; font-weight: 700; line-height: 1.2;}
.tarjeta .nota {color: #6B7C8C; font-size: .85rem;}
[data-testid="stSidebar"] {background: #EEF2F5; border-right: 4px solid #E8A845;}
</style>""", unsafe_allow_html=True)

@st.cache_data                # lee los archivos una sola vez
def cargar_datos():
    predicciones = pd.read_csv("predicciones.csv")
    predicciones[["espesor", "cinta"]] = predicciones[["espesor", "cinta"]].astype(int)   # 40.0 -> 40
    return predicciones, pd.read_csv("originales.csv", index_col="id")

def tarjeta(titulo, valor, nota="", color="#E8A845"):
    # Tarjeta con borde de color a la izquierda (HTML simple)
    return (f'<div class="tarjeta" style="border-left: 6px solid {color};">'
            f'<div class="titulo">{titulo}</div><div class="valor">{valor}</div><div class="nota">{nota}</div></div>')

predicciones, originales = cargar_datos()
espesores = sorted(predicciones["espesor"].unique())
cintas = sorted(predicciones["cinta"].unique())

def cercano(valores, x):
    # Valor de la grilla más cercano al valor real del defecto
    return min(valores, key=lambda v: abs(v - x))

# ---------- Encabezado ----------
st.markdown('<div class="hero"><h1>🔩 Simulador what-if · Riesgo de K_Scatch</h1>'
            '<p>Planchas de acero · Modelo TabPFN (F1 macro test 0,574) · '
            'Felipe Alonso, Sofía Fariña, Vicente Lagunas</p></div>', unsafe_allow_html=True)

# ---------- Controles en la barra lateral ----------
st.sidebar.markdown("## ⚙️ Variables del proceso")
ids = list(originales.index)
id_defecto = st.sidebar.selectbox("Defecto real del test (id)", ids, index=ids.index(11873))
real = originales.loc[id_defecto]
st.sidebar.caption(f"Valores reales: {real['espesor']} mm · cinta {real['cinta']} · {real['acero']}. "
                   "Cada control parte en el valor más cercano.")
espesor = st.sidebar.select_slider("Espesor de la plancha (mm)", espesores, value=cercano(espesores, real["espesor"]))
# La planta tiene dos líneas: la corta (cinta ~1.360) y la larga (cinta ~1.690); casi no hay valores intermedios
linea = st.sidebar.radio("Línea de producción", ["corta", "larga"], index=0 if real["cinta"] < 1500 else 1,
                         horizontal=True, help="Línea corta: cinta ~1.360 · línea larga: cinta ~1.690")
cinta = 1360 if linea == "corta" else 1690                  # valor típico de cada línea
acero = st.sidebar.radio("Tipo de acero", ["A300", "A400"], index=0 if real["acero"] == "A300" else 1, horizontal=True)
st.sidebar.divider()
st.sidebar.caption("🟢 bajo < 18,7 % · 🟠 medio 18,7–50 % · 🔴 alto ≥ 50 %")

# ---------- Probabilidades del escenario elegido vs. defecto original ----------
fila = predicciones[(predicciones["id"] == id_defecto) & (predicciones["espesor"] == espesor) &
                    (predicciones["cinta"] == cinta) & (predicciones["acero"] == acero)].iloc[0]
prob = fila[defectos].astype(float).values / 100
p_critica = prob[defectos.index(critica)] * 100
p_original = real[critica]
clase_predicha = defectos[prob.argmax()]

# Decisión según el riesgo calculado por el modelo
if p_critica >= 50:
    nivel, color, accion = "ALTO", "#C0392B", "Entra al ranking de inspección a fondo del día (reproceso si se confirma)."
elif p_critica >= riesgo_promedio:
    nivel, color, accion = "MEDIO", "#E67E22", "Sobre el promedio: inspeccionar si queda capacidad en el día."
else:
    nivel, color, accion = "BAJO", "#2E8B57", "Tratamiento estándar según su clase predicha."

# ---------- Fila de tarjetas ----------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(tarjeta("Prob. de K_Scatch", f"{p_critica:.1f} %", f"Promedio de la planta: {riesgo_promedio} %", color), unsafe_allow_html=True)
c2.markdown(tarjeta("Cambio vs. original", f"{p_critica - p_original:+.1f} pts", f"Original: {p_original:.1f} %", "#4A6274"), unsafe_allow_html=True)
c3.markdown(tarjeta("Clase más probable", clase_predicha, f"{prob.max() * 100:.0f} % de probabilidad", "#4A6274"), unsafe_allow_html=True)
c4.markdown(tarjeta("Nivel de riesgo", nivel, "Según el umbral del plan", color), unsafe_allow_html=True)
st.write("")

# ---------- Medidor (dona) y gráfico de barras ----------
izq, centro, der = st.columns([1.1, 1.6, 1.3])

with izq.container(border=True):
    st.markdown("**Riesgo de K_Scatch**")
    dona = pd.DataFrame({"parte": ["riesgo", "resto"], "valor": [p_critica, 100 - p_critica]})
    arco = alt.Chart(dona).mark_arc(innerRadius=70, outerRadius=95, cornerRadius=4).encode(
        theta="valor:Q", color=alt.Color("parte:N", scale=alt.Scale(domain=["riesgo", "resto"], range=[color, "#E3E8ED"]), legend=None))
    texto = alt.Chart(pd.DataFrame({"t": [f"{p_critica:.0f} %"]})).mark_text(fontSize=34, fontWeight="bold", color="#1F2D3A").encode(text="t:N")
    st.altair_chart((arco + texto).properties(height=230), width="stretch")

with centro.container(border=True):
    st.markdown("**Probabilidad por tipo de defecto**")
    datos = pd.DataFrame({"Defecto": defectos, "Probabilidad (%)": (prob * 100).round(1)})
    barras = alt.Chart(datos).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("Probabilidad (%):Q", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Defecto:N", sort="-x", title=None),
        color=alt.condition(alt.datum.Defecto == critica, alt.value("#E8A845"), alt.value("#9AABB8")),
        tooltip=["Defecto", "Probabilidad (%)"])
    etiquetas = barras.mark_text(align="left", dx=4, color="#1F2D3A").encode(text=alt.Text("Probabilidad (%):Q", format=".1f"))
    st.altair_chart((barras + etiquetas).properties(height=230), width="stretch")

with der.container(border=True):
    st.markdown("**Recomendación**")
    st.markdown(f'<div style="background:{color}15; border-left:5px solid {color}; padding:.7rem 1rem; border-radius:8px;">'
                f'<b style="color:{color};">Riesgo {nivel}</b><br>{accion}</div>', unsafe_allow_html=True)
    st.markdown(f"Plancha de **{espesor} mm** · acero **{acero}** · línea **{linea}**")
    en_ventana = espesor <= 50 and acero == "A400" and linea == "corta"
    if en_ventana:
        st.markdown("⚠️ **Está dentro de la ventana de mayor riesgo.**")

# ---------- Ventana de operación (resultado del bloque 15) ----------
with st.expander("📌 Ventana de operación (bloque 15 del notebook)", expanded=False):
    st.markdown("Los K_Scatch se concentran en planchas **≤ 50 mm**, acero **A400**, **línea corta**: "
                "**42,8 %** de los defectos son K_Scatch, contra **18,7 %** de promedio. "
                "El modelo estima 43,1 %, casi igual.")
    st.caption("Es una asociación en defectos ya detectados: cambiar de línea es una hipótesis para un piloto, no una causa probada.")

st.caption("Probabilidades calculadas con el mismo modelo TabPFN del notebook para una grilla de escenarios "
           "(60 defectos del test · 11 espesores · 2 líneas · 2 aceros).")
