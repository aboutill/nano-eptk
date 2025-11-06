import pandas as pd
import numpy as np

from scipy.stats import chi2


def convert_pvalue_to_asterisks(p):
    if p < 0.0001:
        return '****'
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


def lme_coefficient_determination(
        model, 
        df,
    ):
    
    # Extract model variance
    var_resid = model.scale
    var_random_effect = float(model.cov_re.iloc[0])
    var_fixed_effect = model.predict(df).var()
    
    # Compute marginal and conditional coefficients of determination
    total_var = var_fixed_effect + var_random_effect + var_resid
    marginal_r2 = var_fixed_effect / total_var
    conditional_r2 = (var_fixed_effect + var_random_effect) / total_var
    
    return marginal_r2, conditional_r2


def lme_confidence_interval(
        model, 
        df, 
        x_col, 
        xlim, 
        n=100, 
        intercept="Intercept",
    ):
    
    # Create dummy predictor array
    pred_x = np.linspace(xlim[0], xlim[1], num=n)
    pred_df = pd.DataFrame({
        x_col: pred_x,
        intercept: 1
    })

    # Fixed effects prediction
    fe_params = model.fe_params
    pred_df["Predicted"] = fe_params[intercept] + fe_params[x_col] * pred_df[x_col]

    # Confidence interval (fixed effects only)
    X = pred_df[[intercept, x_col]]
    cov = model.cov_params().loc[[intercept, x_col], [intercept, x_col]]
    pred_se = np.sqrt(np.sum(np.dot(X, cov) * X, axis=1))
    pred_df["CI_lower"] = pred_df["Predicted"] - 1.96 * pred_se
    pred_df["CI_upper"] = pred_df["Predicted"] + 1.96 * pred_se
    
    return pred_df


def likelihood_ratio(lme_full, lme_restr):
    
    # Extrcat log-likelihood
    llf_full = lme_full.llf
    llf_restr = lme_restr.llf
    
    # Extract degree of freedom
    df_full = lme_full.df_resid 
    df_restr = lme_restr.df_resid 
    
    # Chi squarred test
    lrdf = (df_restr - df_full)
    lrstat = -2*(llf_restr - llf_full)
    lr_pvalue = 1 - chi2(lrdf).cdf(lrstat)
    
    return lrdf, lrstat, lr_pvalue 
