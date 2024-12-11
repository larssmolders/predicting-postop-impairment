import numpy as np
from sklearn.metrics import roc_auc_score
import shap

# assess model performance with average auc across cross validation loops
def calculate_auc(data, model, feature_names, target_name, num_cval_loops, train_percent):
    features = data[feature_names].to_numpy()
    targets = data[target_name].to_numpy().flatten()

    # keep track of class indices to stratify training/testing splits
    class_0_indices = np.argwhere(targets == 0).flatten()
    class_0_n = len(class_0_indices)
    class_1_indices = np.argwhere(targets == 1).flatten()
    class_1_n = len(class_1_indices)

    aucs = []
    for cval_loop in range(num_cval_loops):
        # randomize train/test split for each class separately
        class_0_permutation = np.random.permutation(class_0_n)
        class_1_permutation = np.random.permutation(class_1_n)
        class_0_train_indices = class_0_indices[class_0_permutation][:int(train_percent*class_0_n)]
        class_0_test_indices = class_0_indices[class_0_permutation][int(train_percent*class_0_n):]
        class_1_train_indices = class_1_indices[class_1_permutation][:int(train_percent*class_1_n)]
        class_1_test_indices = class_1_indices[class_1_permutation][int(train_percent*class_1_n):]

        train_indices = np.union1d(class_0_train_indices, class_1_train_indices)
        test_indices = np.union1d(class_0_test_indices, class_1_test_indices)

        # fit and assess performance of this cross validation loop
        fitted_model = model.fit(features[train_indices], targets[train_indices])
        prediction = fitted_model.predict_proba(features[test_indices])[:, 1]
        aucs.append(roc_auc_score(targets[test_indices], prediction))

    return np.average(aucs)

# assess statistical significance of predictor performance with permutation testing
def permutation_testing(data, model, feature_names, target_name, num_cval_loops, train_percent, num_permutations,
                        verbose=False):
    # calculate the auc of the actual predictor
    true_auc = calculate_auc(data, model, feature_names, target_name, num_cval_loops*10, train_percent)
    print(true_auc)
    all_aucs = np.zeros(num_permutations)
    for i in range(num_permutations):
        # randomly shuffle targets and calculate auc on randomized data
        permuted_data = data.copy(deep=True)
        permuted_data[target_name] = np.random.permutation(permuted_data[target_name])
        perm_auc = calculate_auc(permuted_data, model, feature_names, target_name, num_cval_loops, train_percent)

        all_aucs[i] = perm_auc

        if verbose:
            print("%s, %s" % (perm_auc, len(np.argwhere(all_aucs > true_auc)) / (i + 1)))

    # return the predictor performance and the p-value, which is the proportion of randomized predictors that outperform the actual predictor
    return true_auc, len(np.argwhere(all_aucs > true_auc)) / num_permutations

# assess predictor importance with shapley for trees
def shap_scores(data, model, feature_names, target_name):
    fitted_model = model.fit(data[feature_names], data[target_name])
    explainer = shap.TreeExplainer(fitted_model)
    shap_values = explainer(data[feature_names])

    return shap_values
