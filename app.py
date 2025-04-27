import streamlit as st
import pandas as pd
import arff
import io
import matplotlib.pyplot as plt
st.set_page_config(page_title="Data Mining App", layout="wide")

st.title("Data Mining Interface")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (.csv or .arff)", type=["csv", "arff"])

if uploaded_file is not None:
    filename = uploaded_file.name
    if filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.success(" CSV file loaded successfully ✅")
    elif filename.endswith('.arff'):
        decoded = uploaded_file.read().decode('utf-8')
        data = arff.load(io.StringIO(decoded))
        df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
        st.success(" ARFF file loaded successfully ✅")
    else:
        st.error("Unsupported file type.")
        df = None

    # … (imports + upload identiques) …

if 'df' in locals() and df is not None:
    # --- 1) Aperçu du jeu de données --------------------------
    with st.expander("Aperçu des données", expanded=True):
        if len(df) > 10:
            st.dataframe(pd.concat([df.head(5), df.tail(5)]))
        else:
            st.dataframe(df)

    # --- 2) Détails des attributs -----------------------------
    st.markdown("Détails des attributs")
    for col in df.columns:
            with st.expander(f"🔹 {col}"):
                col_type = df[col].dtype
                st.markdown(f"**Type :** `{col_type}`")
                st.markdown(f"**Valeurs distinctes :** {df[col].nunique()}")
                if df[col].nunique() <= 10:
                    st.markdown(f"`{df[col].unique()}`")
                else:
                    st.code(df[col].unique(), language='python')

    # --- 3) Résumé en 5 nombres clés --------------------------
    st.markdown("les 5 nombres clés")
    with st.expander("number summary"):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if numeric_cols.empty:
            st.info("Aucun attribut numérique.")
        else:
            col_5num = st.selectbox("Choisissez un attribut :", numeric_cols, key="five_num")
            stats = {
                "Minimum": df[col_5num].min(),
                "Q1": df[col_5num].quantile(0.25),
                "Médiane": df[col_5num].median(),
                "Q3": df[col_5num].quantile(0.75),
                "Maximum": df[col_5num].max()
            }
            st.dataframe(pd.DataFrame(stats, index=[col_5num]).T)

    # --- 4) Mode ---------------------------------------------
    st.markdown("le mode d'un attribut")
    with st.expander("Mode"):
        col_mode = st.selectbox("Choisissez un attribut :", df.columns, key="mode")
        modes = df[col_mode].mode()
        if modes.empty:
            st.warning("Pas de mode trouvé.")
        else:
            st.write("Mode(s) :")
            for v in modes:
                st.write(f"- {v}")

    # --- 5) Boxplot ------------------------------------------
    st.markdown("Le boxplot (boîte à moustache)")
    with st.expander("Boxplot"):
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if num_cols.empty:
            st.info("Aucun attribut numérique.")
        else:
            col_box = st.selectbox("Attribut numérique :", num_cols, key="box")
            fig, ax = plt.subplots()
            ax.boxplot(df[col_box].dropna())
            ax.set_title(f"Boxplot de '{col_box}'")
            st.pyplot(fig)

    # --- 6) Scatter plot (corrélation) -----------------------
    st.markdown("Le Nuage de points")
    with st.expander("corrélation"):
        num_cols_scatter = df.select_dtypes(include=['int64', 'float64']).columns

        if len(num_cols_scatter) < 2:
            st.info("Au moins deux attributs numériques sont nécessaires pour un scatter plot.")
        else:
            col_x = st.selectbox("Axe X :", num_cols_scatter, key="scatter_x")
            # Retirer col_x de la liste pour éviter X=Y par défaut
            remaining = [c for c in num_cols_scatter if c != col_x]
            col_y = st.selectbox("Axe Y :", remaining, key="scatter_y")

            fig, ax = plt.subplots()
            ax.scatter(df[col_x], df[col_y])
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.set_title(f"Nuage de points : {col_x} vs {col_y}")

            st.pyplot(fig)

    # --- 7) Valeurs manquantes --------------------------------
    st.markdown("Identification des valeurs manquantes")
    with st.expander("Valeurs manquantes"):
        # 1. Tableau récapitulatif
        na_counts = df.isna().sum()
        na_perc = (na_counts / len(df) * 100).round(2)
        na_summary = pd.DataFrame({
            "Nb manquants": na_counts,
            "% manquants": na_perc
        })
        # Garder uniquement les colonnes où il y a des NaN
        na_summary = na_summary[na_summary["Nb manquants"] > 0]

        if na_summary.empty:
            st.success("Aucune valeur manquante dans le dataset ✅")
        else:
            st.subheader("Colonnes avec valeurs manquantes")
            st.dataframe(na_summary)

            # 2. Sélection d’une colonne pour explorer les lignes avec NaN
            col_na = st.selectbox(
                "Choisissez une colonne pour afficher les lignes contenant des NaN :",
                na_summary.index,
                key="na_column"
            )

            if col_na:
                st.markdown(f"### Aperçu des lignes où `{col_na}` est manquant")
                st.dataframe(df[df[col_na].isna()].head(100))  # on limite l’affichage

            # 3. Préparer un futur traitement
            st.markdown("---")
            st.markdown("#### Traitement rapide (optionnel)")
            action = st.radio(
                "Que voulez‑vous faire sur la colonne sélectionnée ?",
                ("Ne rien faire pour l’instant", "Remplir avec la médiane/valeur la plus fréquente", "Supprimer les lignes contenant NaN"),
                key="na_action"
            )

            if st.button("Appliquer l’action"):
                if action == "Remplir avec la médiane/valeur la plus fréquente":
                    if df[col_na].dtype in ["int64", "float64"]:
                        fill_val = df[col_na].median()
                    else:
                        fill_val = df[col_na].mode().iloc[0]
                    df[col_na].fillna(fill_val, inplace=True)
                    st.success(f"Les NaN de `{col_na}` ont été remplacés par `{fill_val}`.")
                elif action == "Supprimer les lignes contenant NaN":
                    df.dropna(subset=[col_na], inplace=True)
                    st.success(f"Lignes contenant un NaN dans `{col_na}` supprimées.")
                st.experimental_rerun()

