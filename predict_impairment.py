import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import shap

from slant_funcs import DMN_indices, FPN_indices, thalamus_indices, DMN_names, FPN_names, thalamus_names
from model_testing import calculate_auc, permutation_testing, shap_scores

# sheet with baseline variables and cognitive z-scores (.xlsx)
# change to pd.read_csv if data is in .csv format
data_sheet = "/path/to/datasheet"
predictor_data = pd.read_excel(data_sheet)
# change this to the column containing patient IDs
patient_id_colname = "pat_ID"
predictor_data = predictor_data.set_index(patient_id_colname)

# list of patient names
data_path = "path/to/data"
pat_names = os.listdir(data_path)

# store DMN, FPN and thalamus degrees for each patient
T0_DMN_degrees = np.zeros((len(pat_names), len(DMN_names)))
T3_DMN_degrees = np.zeros((len(pat_names), len(DMN_names)))
T0_FPN_degrees = np.zeros((len(pat_names), len(FPN_names)))
T3_FPN_degrees = np.zeros((len(pat_names), len(FPN_names)))
T0_thm_degrees = np.zeros((len(pat_names), len(thalamus_names)))
T3_thm_degrees = np.zeros((len(pat_names), len(thalamus_names)))

# calculate SC degrees for each patient
for i, pat in enumerate(pat_names):
    # load pre and post matrices
    pre_mat = np.load(os.path.join(data_path, str(pat), "pre", "tracts",
                                   "structural_matrix_slant_4m.npy"))
    post_mat = np.load(os.path.join(data_path, str(pat), "post", "tracts",
                                   "structural_matrix_slant_4m.npy"))

    # degrees are the sum of weights from a ROI to all other ROIs
    T0_DMN_degrees[i] = np.sum(pre_mat[DMN_indices], axis=1)
    T3_DMN_degrees[i] = np.sum(post_mat[DMN_indices], axis=1)
    T0_FPN_degrees[i] = np.sum(pre_mat[FPN_indices], axis=1)
    T3_FPN_degrees[i] = np.sum(post_mat[FPN_indices], axis=1)
    T0_thm_degrees[i] = np.sum(pre_mat[thalamus_indices], axis=1)
    T3_thm_degrees[i] = np.sum(post_mat[thalamus_indices], axis=1)

# name the variables
T0_DMN_names = ["T0_" + x for x in DMN_names]
T3_DMN_names = ["T3_" + x for x in DMN_names]
T0_FPN_names = ["T0_" + x for x in FPN_names]
T3_FPN_names = ["T3_" + x for x in FPN_names]
T0_thm_names = ["T0_" + x for x in thalamus_names]
T3_thm_names = ["T3_" + x for x in thalamus_names]

# store the connectivity data in a dataframe
degree_dataframe = pd.DataFrame(data=np.hstack((np.array(pat_names).reshape((-1, 1)), T0_DMN_degrees, T0_FPN_degrees,
                                                T0_thm_degrees, T3_DMN_degrees, T3_FPN_degrees, T3_thm_degrees)),
                                columns= ["pat_ID"] + T0_DMN_names + T0_FPN_names + T0_thm_names + T3_DMN_names + \
                                         T3_FPN_names + T3_thm_names)
degree_dataframe.set_index('pat_ID')

# join the dataframes to have baseline, connectivity and cognitive scores in a single sheet
all_data = pd.merge(predictor_data, degree_dataframe, on='pat_ID')


# names of columns of baseline variables available at T0 without cognitive testing
T0_covariate_names = ['age', 'tumor volume', 'education', 'sex', 'loc L', 'loc R', 'loc_front', 'loc_par', 'loc_temp',
                      'loc_occ', 'loc_ins']

# names of columns of baseline variables available at T0
covariates_available_at_T0 = ['age', 'tumor volume', 'education', 'sex', 'loc L', 'loc R', 'loc_front', 'loc_par',
                              'loc_temp', 'loc_occ', 'loc_ins', "T0_2imp"]

# names of columns of baseline variables available at T3
T3_covariate_names = T0_covariate_names + ['RT', 'CT', 'T0_2imp']

# readable names for variables
T0_covariate_screen_names = ["age", "tumor volume", "education level", "sex (female)",
                             "left hemisphere location", "right hemisphere location", "frontal lobe location",
                             "parietal lobe location", "temporal lobe location", "occipital lobe location",
                             "insular location"]
covariates_available_at_T0_screen_names = ["age", "tumor volume", "education level", "sex (female)",
                                           "left hemisphere location", "right hemisphere location",
                                           "frontal lobe location", "parietal lobe location", "temporal lobe location",
                                           "occipital lobe location", "insular location", "impairment at T0"]
T3_covariate_screen_names = T0_covariate_screen_names + ["radiotherapy between T0 and T3",
                                                         "chemotherapy between T0 and T3",
                                                         "impairment at T0"]

# define sets of predictors for each outcome
T0_predictor_sets = [
    T0_covariate_names,
    T0_DMN_names + T0_thm_names,
    T0_FPN_names + T0_thm_names,
    T0_DMN_names + T0_covariate_names + T0_thm_names,
    T0_FPN_names + T0_covariate_names + T0_thm_names,
    T0_DMN_names + T0_FPN_names + T0_covariate_names + T0_thm_names,
]

T0_predictor_set_screen_names = [
    T0_covariate_screen_names,
    DMN_names + thalamus_names,
    FPN_names + thalamus_names,
    DMN_names + T0_covariate_screen_names + thalamus_names,
    FPN_names + T0_covariate_screen_names + thalamus_names,
    DMN_names + FPN_names + T0_covariate_screen_names + thalamus_names,
]


available_at_T0_predictor_sets = [
    covariates_available_at_T0,
    T0_DMN_names + T0_thm_names,
    T0_FPN_names + T0_thm_names,
    T0_DMN_names + covariates_available_at_T0 + T0_thm_names,
    T0_FPN_names + covariates_available_at_T0 + T0_thm_names,
    T0_DMN_names + T0_FPN_names + covariates_available_at_T0 + T0_thm_names,
]

available_at_T0_predictor_screen_names = [
    covariates_available_at_T0_screen_names,
    DMN_names + thalamus_names,
    FPN_names + thalamus_names,
    DMN_names + covariates_available_at_T0_screen_names + thalamus_names,
    FPN_names + covariates_available_at_T0_screen_names + thalamus_names,
    DMN_names + FPN_names + covariates_available_at_T0_screen_names + thalamus_names,
]


T3_predictor_sets = [
    T3_covariate_names,
    T3_DMN_names + T3_thm_names,
    T3_FPN_names + T3_thm_names,
    T3_DMN_names + T3_covariate_names + T3_thm_names,
    T3_FPN_names + T3_covariate_names + T3_thm_names,
    T3_DMN_names + T3_FPN_names + T3_covariate_names + T3_thm_names,
]

T3_predictor_set_screen_names = [
    T3_covariate_screen_names,
    DMN_names + thalamus_names,
    FPN_names + thalamus_names,
    DMN_names + T3_covariate_screen_names + thalamus_names,
    FPN_names + T3_covariate_screen_names + thalamus_names,
    DMN_names + FPN_names + T3_covariate_screen_names + thalamus_names,
]

# the model is a random forest with maximum depth 3
model = RandomForestClassifier(max_depth=3)

# choose the specific predictors and outcomes for this run
predictor_set = T0_predictor_sets
screen_name_set = T0_predictor_set_screen_names
target_name = "T0_2imp"


for i, predictors in enumerate(predictor_set):
    # predictor performance is the average auc over cross validation loops
    print(calculate_auc(all_data, model, predictors, target_name, num_cval_loops=100, train_percent=.75))

    # statistical significance is assessed with permutation testing
    print(permutation_testing(all_data, model, predictors, target_name, num_cval_loops=100, train_percent=.75,
                              num_permutations=1000, verbose=False))

    # plot predictor importances with shaply for trees
    shap_values = shap_scores(all_data, model, predictors, target_name)
    shap.plots.violin(shap_values[:, :, 1], show=False, max_display=5, feature_names=screen_name_set[i])
    plt.tight_layout()
    plt.show()
    print("--------")
