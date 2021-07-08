import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title='NanoInformatics Portal - CMS Zewail City', page_icon=None, layout='centered', initial_sidebar_state='auto')

st.write("""
![Nanoscience Program Logo](https://i.ibb.co/fQzXDrX/Core-Logo-copy.png)

# The NanoInformatics Portal of Center for Materials Science (CMS) at Zewail City (ZC)
## The first NanoInfomatics portal in Egypt and the MENA region.

This app predicts the **Toxicity** of nanomaterials given some parameters using supervised **Machine Learning** models. Models are built based upon the Nanomaterials-Biological Interactions Knowledgebase by Oregon State University linked [here](http://nbi.oregonstate.edu/analysis.php).
""")

# def func():
#    import webbrowser
#    webbrowser.open("https://google.com")

# st.button("Contribute a dataset", on_click=func())
# Sidebar

df = pd.read_csv('data-numeric.csv')

st.sidebar.header("Model Selection and Tuning")

mdl = st.sidebar.selectbox('Model', ('SVM Classifier', 'Random Forest Regression'))
if mdl == 'SVM Classifier':
    from sklearn.svm import SVC
    df_target = pd.read_csv('full-materials-data.csv')['Toxicity']

    for i in range(len(df_target)):
        if df_target[i] < 0.5:
            df_target[i] = 1 # nontoxic
        else:
            df_target[i] = 2 # toxic

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(df, df_target, test_size=0.3,random_state=101)

    # Model training
    c_var = st.sidebar.select_slider('C Parameter', [0.1, 1, 10, 100, 1000], 1)
    g_var = st.sidebar.select_slider('Gamma Parameter', [1, 0.1, 0.01, 0.001, 0.0001], 0.01)
    k_var = st.sidebar.selectbox('Kernel', ['rbf', 'linear'])
    model = SVC(C = c_var, gamma = g_var, kernel = k_var, probability=True)
    model.fit(X_train, y_train)
else:
    df_target = pd.read_csv('full-materials-data.csv')['Toxicity']
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(df, df_target, test_size=0.2,random_state=109)
    n_est = st.sidebar.select_slider('Number of Estimators', [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], 1000)
    max_d = st.sidebar.select_slider('Max Depth', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None], 110)
    max_f = st.sidebar.selectbox('Max Features', ['sqrt', 'auto'])
    msl = st.sidebar.select_slider('Min Samples Leaf', [1, 2, 4], 1)
    mss = st.sidebar.select_slider('Min Samples Split', [2, 5, 10], 2)
    btstrp = st.sidebar.selectbox('Bootstrap', [True, False], True)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, max_features=max_f, min_samples_leaf=msl, min_samples_split=mss, bootstrap=btstrp, random_state = 42)
    model.fit(X_train, y_train)


# Main Page
st.header("User Input")
family_dict = {'carbon': 7,
 'cellulose': 5,
 'dendrimer': 2,
 'metal': 1,
 'metal oxide': 3,
 'polymeric': 4,
 'semiconductor': 6}

core_dict = {'1,4-diaminobutane [DAB]': 2,
 'aluminum oxide [Al2O3]': 3,
 'antimony (iii) oxide [sb2o3]': 26,
 'carbon [c60]': 21,
 'carbon [c]': 29,
 'cellulose': 14,
 'cerium oxide [CeO2]': 6,
 'copper oxide [CuO]': 27,
 'dysprosium oxide [Dy2O3]': 8,
 'erbium oxide [Er2O3]': 12,
 'europium (iii) oxide [eu2o3]': 22,
 'europium oxide [eu2o3]': 24,
 'gadolinium oxide [Gd2O3]': 7,
 'gold [Au]': 1,
 'holmium oxide [Ho2O3]': 10,
 'iron oxide [Fe2O3]': 18,
 'lead sulfide [PbS]': 19,
 'neodymium oxide [nd2o3]': 25,
 'platinum [pt]': 23,
 'polystyrene': 13,
 'samarium oxide [Sm2O3]': 11,
 'silica [si]': 28,
 'silicon dioxide [SiO2]': 16,
 'silver [Ag]': 17,
 'terbium (iii,iv) oxide [tb4o7]': 30,
 'thiophosphoryl-pmmh': 20,
 'titanium dioxide [TiO2]': 4,
 'yttrium oxide [Y2O3]': 9,
 'zinc oxide [ZnO]': 15,
 'zirconium oxide [O2Zr]': 5}

additional_dict = {'gold [Au]': 3,
 'manganese [mn]': 4,
 'N/A': 1,
 'silicon dioxide [SiO2]': 2,
 'zinc oxide [ZnO]': 5}

surface_dict = {'2,2,2-[mercaptoethoxy(ethoxy)]ethanol [MEEE]': 5,
 '2,3-dimercaptopropanesulfonic acid, sodium salt': 22,
 '2-(2-mercaptoethoxy)ethanol [MEE]': 1,
 '2-,2-aminoethoxyethanol [aee]': 31,
 '2-mercaptoethanesulfonate [MES]': 3,
 '2-mercaptoethylphosphonic acid [MEPA]': 6,
 '3-mercaptopropanesulfonic acid, sodium salt': 21,
 '6-mercaptohexanoic acid [MHA]': 4,
 'N,N,N-trimethylammoniumethanethiol [TMAT]': 2,
 'aldehyde': 28,
 'aldehyde-sulfate': 11,
 'amidoethanol': 9,
 'amine': 7,
 'ascorbic acid': 27,
 'benzoic acid': 17,
 'carboxyl': 23,
 'citrate': 18,
 'cyclohexane carboxilic acid': 16,
 'dichlorophosphinothioyl': 29,
 'ethylenediamine': 25,
 'ferric oxide': 20,
 'hexamethylene diamine': 26,
 'hydroxypropyl-trimethylammonium chloride': 32,
 'octanoic acid': 14,
 'oleic acid': 13,
 'para-nitrobenzoic acid': 15,
 'phosphate': 19,
 'phosphatidylcholine': 12,
 'rhodamine': 30,
 'silanol': 33,
 'succinamic acid': 8,
 'taurine': 24,
 'unknown': 10}

shape_dict = {'cylindrical': 3,
 'dendritic': 7,
 'fibrous': 6,
 'irregular-angular': 4,
 'regular-angular': 8,
 'rod': 5,
 'spherical': 1,
 'unknown': 2}

charge_dict = {'+': 2, '-': 3, '0': 1}

def user_input_features():
    family = st.selectbox('Family',('metal', 'dendrimer', 'metal oxide', 'polymeric', 'cellulose',
       'semiconductor', 'carbon'))
    core = st.selectbox('Core',('gold [Au]', '1,4-diaminobutane [DAB]', 'aluminum oxide [Al2O3]',
       'titanium dioxide [TiO2]', 'zirconium oxide [O2Zr]',
       'cerium oxide [CeO2]', 'gadolinium oxide [Gd2O3]',
       'dysprosium oxide [Dy2O3]', 'yttrium oxide [Y2O3]',
       'holmium oxide [Ho2O3]', 'samarium oxide [Sm2O3]',
       'erbium oxide [Er2O3]', 'polystyrene', 'cellulose',
       'zinc oxide [ZnO]', 'silicon dioxide [SiO2]', 'silver [Ag]',
       'iron oxide [Fe2O3]', 'lead sulfide [PbS]', 'thiophosphoryl-pmmh',
       'carbon [c60]', 'europium (iii) oxide [eu2o3]', 'platinum [pt]',
       'europium oxide [eu2o3]', 'neodymium oxide [nd2o3]',
       'antimony (iii) oxide [sb2o3]', 'copper oxide [CuO]',
       'silica [si]', 'carbon [c]', 'terbium (iii,iv) oxide [tb4o7]'))
    additional = st.selectbox('Additional to core', ('N/A', 'silicon dioxide [SiO2]', 'gold [Au]', 'manganese [mn]',
       'zinc oxide [ZnO]'))
    surface = st.selectbox('Surface Chemistry', ('2-(2-mercaptoethoxy)ethanol [MEE]',
       'N,N,N-trimethylammoniumethanethiol [TMAT]',
       '2-mercaptoethanesulfonate [MES]', '6-mercaptohexanoic acid [MHA]',
       '2,2,2-[mercaptoethoxy(ethoxy)]ethanol [MEEE]',
       '2-mercaptoethylphosphonic acid [MEPA]', 'amine',
       'succinamic acid', 'amidoethanol', 'unknown', 'aldehyde-sulfate',
       'phosphatidylcholine', 'oleic acid', 'octanoic acid',
       'para-nitrobenzoic acid', 'cyclohexane carboxilic acid',
       'benzoic acid', 'citrate', 'phosphate', 'ferric oxide',
       '3-mercaptopropanesulfonic acid, sodium salt',
       '2,3-dimercaptopropanesulfonic acid, sodium salt', 'carboxyl',
       'taurine', 'ethylenediamine', 'hexamethylene diamine',
       'ascorbic acid', 'aldehyde', 'dichlorophosphinothioyl',
       'rhodamine', '2-,2-aminoethoxyethanol [aee]',
       'hydroxypropyl-trimethylammonium chloride', 'silanol'))
    shape = st.selectbox('Shape', ('spherical', 'unknown', 'cylindrical', 'irregular-angular', 'rod',
       'fibrous', 'dendritic', 'regular-angular'))
    charge = st.selectbox('Charge', ('0', '+', '-'))
    size = st.number_input('Size', 0., 125.)
    st.markdown("""Note: allowed size is from 0 to 125.""")
    dose = st.number_input('Dose', 0.000000013, 0.00025)
    st.write('The entered dose is', dose, '.    ')
    st.markdown("""Note: allowed dose is from 0.000000013 (13 ppb) to 0.00025 (250 ppm).""")

    data = {'Family': family_dict[family],
            'Core': core_dict[core],
            'Additional': additional_dict[additional],
            'Surface Chemistry': surface_dict[surface],
            'Shape': shape_dict[shape],
            'Size': size,
            'Charge': charge_dict[charge],
            'Dose': dose}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

file = st.file_uploader("Or upload your input CSV file for multiple predictions", type=["csv"])
st.markdown("""
[Example CSV input file](./cms-nanoinf-example.csv)
""")

if file is not None:
    df = pd.read_csv(file)
else:
    df = input_df

b_result = st.button("Predict")

if b_result == True:
    st.header("Predictions")
    # Displays the user input features
    st.subheader('User Input features')

    if file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        st.write(df)

    if mdl == "SVM Classifier":
        # Apply model to make predictions
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        st.subheader('Prediction')
        toxicity = {1: "Non-toxic - less than LD50", 2: "Toxic - above LD50"}
        st.write(toxicity[prediction[0]])
        # st.write(prediction[0])

        st.subheader('Prediction Probability')
        st.write(prediction_proba)

        st.subheader('Model Evaluation Metrics')
        " "
        pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.metrics import confusion_matrix

        acc = accuracy_score(y_test, pred)
        "- This model works with accuracy ", round(acc*100, 4), '%'
        "- The precision is ", 0.92, "."
        "- The recall is ", 0.99, "."

        "- **Confusion Matrix**"
        from sklearn.metrics import plot_confusion_matrix
        plot_confusion_matrix(model, X_test, y_test)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        "- **Receiver Operating Characteristic Curve**"
        from sklearn.metrics import plot_roc_curve
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()

    else:
        # Apply model to make predictions
        prediction = model.predict(df)
        st.subheader('Prediction')
        st.write(round(prediction[0], 5))
        if prediction[0] < 0.5:
            st.write('This shows toxicity less that LD50 which is non-lethal (i.e., non-toxic).')
        else:
            st.write('This shows toxicity higher that LD50 which is lethal (i.e., toxic).')

        st.subheader('Model Evaluation Metrics')
        " "
        from sklearn import metrics
        pred_rf = model.predict(X_test)

        # model evaluation for testing set

        r2 = metrics.r2_score(y_test, pred_rf)
        mae = metrics.mean_absolute_error(y_test, pred_rf)
        mse = metrics.mean_squared_error(y_test, pred_rf)

        '- R2 score is ', round(r2, 5)
        '- MAE is ', round(mae,5)
        '- MSE is ', round(mse, 5)

        "- **Predicted vs. Actual Plot**"
        fig, ax = plt.subplots()
        ax.scatter(pred_rf, y_test, edgecolors=(0, 0, 1))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot()

