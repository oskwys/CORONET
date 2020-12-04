import numpy as np
import matplotlib.pyplot as plt
import os
import joblib



def predict_and_explain(x, model, explainer, plot_expl_barplot = True, path_to_save_plots=''):
    """
    Predicts and explains the prediction using pre-trained model and explainer.
    It outputs two dictionaries: 1) prediction, 2) explanation
    Optionally, it can plot and save a barplot with shap values explaining the prediction.

    Before running the function, the model and the explainer should be loaded.

    Contains admission_threshold and severe_condition_threshold which are specified inside this function.

    Parameters
    ----------
    x : dict
    A dictionary with keys = features, values = patient's parameters values.
    Dictionary format:

    {'Age': value,
     'Total number comorbidities': value,
     'Haematological cancer': value,
     'Cancer stage': value,
     'NEWS2': value,
     'Platelets': value,
     'Albumin': value,
     'CRP': value,
     'Neutrophil': value,
     'Lymphocyte': value}


    model : sklearn predictive model

    explainer : shap.TreeExplainer object

    plot_expl_barplot : bool - default True, if False the function does not generate the barplot with the explanation

    path_to_save_plots : str - directory to save png file with the figure

    Returns
    ------
    prediction : dict
    a dictionary with keys: 'Predicted_score' and 'Recommendation'

    explanation : dict
    a dictionary with shap values for each feature sorted by absolute value


    """
    admission_threshold = 1.05
    severe_condition_threshold = 1.8

    x_trans = transform_x_values(x)

    prediction = get_prediction_for_x(x_trans, model, admission_threshold, severe_condition_threshold)

    explanation = get_shap_values_for_x(x_trans, explainer, sort_explanation=True)

    if plot_expl_barplot:
        plot_local_explanation_shap(explanation, x, path_to_save_plots)

    return prediction, explanation

def load_predictive_model(file_path):
    """
     Loads predictive model stored in a .pkl file from 'file_path'
     The model is a Random Forest model trained using sklearn library and saved to .pkl file using joblib library (using joblib.dump command)
     required libraries:
      joblib

    Parameters:
    -----------
     file_path : str
     Path where the model is stored


    Return
    ------
    model : object sklearn regression model
    """

    model = joblib.load(file_path)

    return model

def load_explainer(file_path):
    """
    Loads explainer stored in a .pkl file from 'file_path'
    The explainer is a Explainer from SHAP library (https://github.com/slundberg/shap)
    created using function shap.TreeExplainer(model), where 'model' is the predictive model used in CORONET
    and saved to .pkl file using joblib library (using joblib.dump command)
    required libraries:
       joblib

    Parameters:
    -----------
     file_path : str
     Path where the explainer is stored
     

    Return
    ------
    explainer : object shap explainer
    """


    explainer = joblib.load(file_path)

    return explainer


def get_prediction_for_x(x, model, admission_threshold, severe_condition_threshold):
    """
    Calculates the score and assigns recommendation based on given thresholds.
    Uses transformed x (with transformed CRP and NLR) and predictive model and calculates the score (range 0.0-3.0),
    It also outputs a string with a recommendation from the list of three:
    - 'consider discharge'
    - 'consider admission'
    - 'high risk of severe condition'


    Parameters:
    -----------
    x : dict
    A dictionary with keys = features, values = patient's parameters values. CRP and NLR values should be transformed.
    Dictionary format:

    {'Age': value,
     'Total number comorbidities': value,
     'Haematological cancer': value, (binary, 0 or 1)
     'Cancer stage 12': value, (binary, 0 or 1)
     'Cancer stage 3': value, (binary, 0 or 1)
     'Cancer stage 4': value, (binary, 0 or 1)
     'NEWS2': value,
     'Platelets': value,
     'Albumin': value,
     'log2_CRP': value,
     'log2_NLR': value}


    model : sklearn predictive model

    admission_threshold : float
    A threshold defined by the researcher.
    Above this value all recommendation will be 'consider admission' or 'high risk of severe condition'.
    Below this value all recommendation will be 'consider discharge'.

    severe_condition_threshold : float
    A threshold defined by the researcher.
    Above this value all recommendation will be 'high risk of severe condition'.
    Below this value all recommendation will be 'consider discharge' or 'consider admission'.

    Return
    ------
    prediction : dict
    a dictionary with predicted score (str, the score rounded to 2 decimals) and textual recommendation (str).
    Dictionary format (example values):
    {'Predicted_score': '0.95',
     'Recommendation': 'consider discharge'}

    """

    x_to_model = np.array(list(x.values())).reshape(1, -1)

    predicted_score = np.round(model.predict(x_to_model)[0], 2)


    recommendations = ['consider discharge', 'consider admission', 'high risk of severe condition']

    if predicted_score < admission_threshold:
        recommendation = recommendations[0]
    elif predicted_score > severe_condition_threshold:
        recommendation = recommendations[2]
    else:
        recommendation = recommendations[1]

    # convert to string with 2 decimal places (for consitency in showing the coronet score to the user)
    predicted_score = f'{predicted_score:.2f}'

    prediction = {'Predicted_score': predicted_score, 'Recommendation': recommendation}

    return prediction


def transform_x_values(x):
    """
    Transforms CRP into log2(CRP)
    Transforms Neutrophil and Lymphocyte into NLR (Neutrophil:Lymphocyte Ratio)
    If Lymphocyte < 0.1, then Lymhpcyte is set to 0.1 before calculating the NLR
    Transformation is needed because the model was trained on transformed values.
    Transformation of the training set was performed due to skewed distribution of these parameters.
    After transformation the distributions of CRP and NLR are similar to normal distribution.

    Parameters:
    ----------
    param x : dict
    x : dict
    A dictionary with keys = features, values = patient's parameters values.
    Dictionary format:

    {'Age': value,
     'Total number comorbidities': value,
     'Haematological cancer': value, (binary, 0 or 1)
     'Cancer stage': value, (0, 1, 2, 3 or 4)
     'NEWS2': value,
     'Platelets': value,
     'Albumin': value,
     'CRP': value,
     'Neutrophil': value,
     'Lymphocyte': value}

    Return:
    ------
    x_transformed : dict
    A dictionary with keys = features, values = patient's parameters values. CRP and Neutrophils are transformed using log2().
    Features 'CRP' are replaced by 'log2_CRP'. Neutrophil and Lyphocyte are replaced by 'log2_NLR'.
    Cancer stage is replaced by Cancer stage 12, Cancer stage 3 and Cancer stage 4.
    Dictionary format:

    {'Age': value,
     'Total number comorbidities': value,
     'Haematological cancer': value, (binary, 0 or 1)
     'Cancer stage 12': value, (binary, 0 or 1)
     'Cancer stage 3': value, (binary, 0 or 1)
     'Cancer stage 4': value, (binary, 0 or 1)
     'NEWS2': value,
     'Platelets': value,
     'Albumin': value,
     'log2_CRP': value,
     'log2_NLR': value}
    """
    x_transformed = x.copy()

    x_transformed['log2_CRP'] = np.log2(x['CRP'] + 1)

    if x_transformed['Lymphocyte'] < 0.1:
        x_transformed['Lymphocyte'] = 0.1

    x_transformed['log2_NLR'] = np.log2(x['Neutrophil']/x_transformed['Lymphocyte'] + 1)

    # cancer stage conversion to dummy variables

    x_transformed['Cancer stage 12'] = 0
    x_transformed['Cancer stage 3'] = 0
    x_transformed['Cancer stage 4'] = 0
    x_transformed['Haematological cancer'] = 0

    if x['Cancer stage'] == 0:
        x_transformed['Haematological cancer'] = 1
    elif (x['Cancer stage'] == 1) | (x['Cancer stage'] == 2):
        x_transformed['Cancer stage 12'] = 1
    elif x['Cancer stage'] == 3:
        x_transformed['Cancer stage 3'] = 1
    elif x['Cancer stage'] == 4:
        x_transformed['Cancer stage 4'] = 1



    del x_transformed['CRP'], x_transformed['Neutrophil'], x_transformed['Lymphocyte'], x_transformed['Cancer stage']

    # Reorder dict after adding cancer stages
    desired_order_list = ['Age',
                          'Total number comorbidities',
                          'Haematological cancer',
                          'Cancer stage 12',
                          'Cancer stage 3',
                          'Cancer stage 4',
                          'NEWS2',
                          'Platelets',
                          'Albumin',
                          'log2_CRP',
                          'log2_NLR']

    x_transformed = {k: x_transformed[k] for k in desired_order_list}

    return x_transformed


def get_shap_values_for_x(x, explainer, sort_explanation=True):
    """
    Computes shapley values of local explanation for 'x'.
    Uses 'explainer' which is an explainer object from shap library.
    Generated 'explanation' can sorted (default) or in the same order as 'x'.

    Parameters:
    ----------
    x : dict
    A dictionary with keys = features, values = patient's parameters values. CRP and NLR values should be transformed.
    Dictionary format:

    {'Age': value,
     'Total number comorbidities': value,
     'Haematological cancer': value, (binary, 0 or 1)
     'Cancer stage 12': value, (binary, 0 or 1)
     'Cancer stage 3': value, (binary, 0 or 1)
     'Cancer stage 4': value, (binary, 0 or 1)
     'NEWS2': value,
     'Platelets': value,
     'Albumin': value,
     'log2_CRP': value,
     'log2_NLR': value}

    explainer : shap.Explainer object

    sort_explanation : bool
    default True, if False the keys of explanation dict will be in the same order as x.
    If True, the explanation dict will be sorted by absolute value of shap value (the highest - most important - are at the bottom)

    Return:
    -------
    explanation : dict
    a dictionary with shap values for each feature sorted by absolute value (sorting is optional but default True)
    Dictionary format:
    {'Age': shap_value,
     'Total number comorbidities': shap_value,
     'Haematological cancer': shap_value,
     'Cancer stage 12': shap_value,
     'Cancer stage 3': shap_value,
     'Cancer stage 4': shap_value,
     'NEWS2': shap_value,
     'Platelets': shap_value,
     'Albumin': shap_value,
     'log2_CRP': shap_value,
     'log2_NLR': shap_value}


    """
    x_to_model = np.array(list(x.values()))

    features = list(x.keys())

    shap_values = np.round(explainer.shap_values(x_to_model), 4)

    explanation = {}

    for i, feature in enumerate(features):
        explanation[feature] = shap_values[i]

    if sort_explanation:
        explanation = {k: v for k, v in sorted(explanation.items(), key=lambda item: np.abs(item[1]), reverse=False)}

    return explanation


def plot_local_explanation_shap(shap_dict, x, path_to_save):
    """
    Plot a red-green barplot showing the contribution of each feature to the prediction.
    The contribution is equal to shap value for given feature.
    Negative shap values contribute to the 'consider discharge' recommendation and are represented as green bars on the left side of the plot.
    Positive shap values contribute to the 'consider admission' or 'high risk of severe condition' recommendation and are represented as red bars on the right side of the plot.

    Next to the bars a value of given parameter is shown in a textbox.

    Important: bar width corresponds to the shap value, not to the parameter value displayed in the textbox.

    Saves the figure as 'local_explanation_shap.png'

    Parameters:
    -----------
    shap_dict : : dict
    a dictionary with shap values for each feature sorted by absolute value (sorting is optional but default True)
    Dictionary format:
    {'Age': shap_value,
     'Total number comorbidities': shap_value,
     'Haematological cancer': shap_value,
     'Cancer stage 12': shap_value,
     'Cancer stage 3': shap_value,
     'Cancer stage 4': shap_value,
     'NEWS2': shap_value,
     'Platelets': shap_value,
     'Albumin': shap_value,
     'log2_CRP': shap_value,
     'log2_NLR': shap_value}


    x : dict
    A dictionary with keys = features, values = patient's parameters values. (note: CRP, Neutrophil and Lymphocyte not transformed)
    Dictionary format:

    {'Age': value,
     'Total number comorbidities': value,
     'Haematological cancer': value,
     'Cancer stage': value,
     'NEWS2': value,
     'Platelets': value,
     'Albumin': value,
     'CRP': value,
     'Neutrophil': value,
     'Lymphocyte': value}


    path_to_save : str
     Directory where the png file with figure should be saved.

    Returns:
    -------


    """
    # remove redundant cancer stages
    if x['Haematological cancer'] == 1:
        # if patient has Haematological cancer, he can't have a solid cancer stage, so any cancer stage should be removed from the explanation
        del shap_dict['Cancer stage 12'], shap_dict['Cancer stage 3'], shap_dict['Cancer stage 4']
    elif (x['Cancer stage'] == 1) | (x['Cancer stage'] == 2):
        # if patient has Cancer stage 1 or 2, importance on not having cancer stage 3 or 4 should be removed. Importance of not having Haem cancer still should be preseneted
        del shap_dict['Cancer stage 3'], shap_dict['Cancer stage 4']
    elif x['Cancer stage'] == 3:
        # if patient has Cancer stage 3, importance on not having cancer stage 1,2 or 4 should be removed. Importance of not having Haem cancer still should be preseneted
        del shap_dict['Cancer stage 12'], shap_dict['Cancer stage 4']
    elif x['Cancer stage'] == 4:
        # if patient has Cancer stage 4, importance on not having cancer stage 1,2 or 3 should be removed. Importance of not having Haem cancer still should be preseneted
        del shap_dict['Cancer stage 12'], shap_dict['Cancer stage 3']

    # sort shap_values dictionary by absolute value
    shap_dict_sorted = shap_dict  # {k: v for k, v in sorted(shap_dict.items(), key=lambda item: np.abs(item[1]), reverse=False)}

    fig, ax = plt.subplots(figsize=(13, 6))

    features = list(shap_dict_sorted.keys())

    values = list(shap_dict_sorted.values())

    # plot barplot
    bars = ax.barh(width=values, y=features, linewidth=1, edgecolor='black')

    # assing bar colors (red for features voting for 'admission', green for features voting for 'discharge')
    for j, bar in enumerate(bars):
        if values[j] < 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
        bar.set_edgecolor('k')

    ax.set_xticklabels([None])

    # add arrows at the top
    props = dict(boxstyle="larrow,pad=0.3", facecolor='white', alpha=1)
    text = 'DISCHARGE'
    ax.text(0.45, 1.05, text, bbox=props, transform=ax.transAxes, va='bottom', ha='right', fontsize=15)
    props = dict(boxstyle="rarrow,pad=0.3", facecolor='white', alpha=1)
    text = 'ADMISSION'
    ax.text(0.55, 1.05, text, bbox=props, transform=ax.transAxes, va='bottom', ha='left', fontsize=15)
    ax.set_yticklabels([None])

    # add bars description (feature name and its value, i.e. real value, not the shap value)
    for m in range(len(values)):
        parameter = features[m]
        shap_value = values[m]

        if parameter == 'log2_NLR':
            # RF model uses transformed Neutrophil, but to show the Neutrophil value on the plot, we need to refer to initial 'x' instead of 'x_transformed'
            unit = ''
            text = 'Neutrophil:Lymphocyte Ratio' + ' = ' + str(
                np.round(x['Neutrophil'] / x['Lymphocyte'], 1)) + ' ' + unit

        elif parameter == 'log2_CRP':
            # RF model uses transformed CRP, but to show the CRP value on the plot, we need to refer to initial 'x' instead of 'x_transformed'
            unit = 'mg/L'
            text = 'C-reactive protein' + ' = ' + str(np.round(x['CRP'], 2)) + ' ' + unit
        elif parameter == 'Albumin':
            unit = 'g/L'
            text = parameter + ' = ' + str(np.round(x[parameter], 2)) + ' ' + unit
        elif parameter == 'Lymphocyte':
            unit = 'g/L'
            text = parameter + ' = ' + str(np.round(x[parameter], 2)) + ' ' + unit
        elif parameter == 'Platelets':
            unit = 'x10^9/L'
            text = parameter + ' = ' + str(np.round(x[parameter], 2)) + ' ' + unit
        elif 'Cancer stage' in parameter:
            if parameter[-1] == '2':
                text = 'Cancer stage 1 or 2'
            else:
                text = 'Cancer stage {}'.format(parameter[-1])
        elif parameter == 'Haematological cancer':
            # if a patient does not have Haem cancer, it means that he has solid cancer
            if x[parameter] == 1:
                text = 'Haematological cancer'
            else:
                text = 'Solid cancer'
        else:
            text = parameter + ' = ' + str(np.int(np.round(x[parameter], 0)))

        if shap_value > 0:
            ha = 'left'
        else:
            ha = 'right'

        ax.text(shap_value + 0.008 * np.sign(shap_value), m, text, ha=ha, va='center', fontsize=18)

    # ax.set_xlim([-np.abs(values).max() - 0.1, np.abs(values).max() + .1])
    ax.set_xlim([-.5, .5])

    # remove axes lines
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    plt.subplots_adjust(top=1.1)
    plt.tight_layout()
    path_to_save = os.path.join(path_to_save, 'local_explanation_shap.png')
    plt.savefig(path_to_save, dpi=400)