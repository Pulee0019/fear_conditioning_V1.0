import os
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, shapiro, wilcoxon, mannwhitneyu, normaltest, anderson, levene
import matplotlib.pyplot as plt


def check_normality(data, alpha=0.05, method='auto'):
    """Check if the data follows a normal distribution using the specified method (Shapiro-Wilk, D'Agostino-Pearson, or Anderson-Darling). Automatically selects method based on sample size if 'auto' is chosen."""
    data = np.array(data)
    n = len(data)
    
    if n < 3:
        print(f"Warning: Sample size ({n}) is too small for normality test. Assuming non-normal.")
        return False, np.nan, "Sample too small", np.nan
    
    if method == 'auto':
        if n <= 50:
            method = 'shapiro'
        elif n <= 5000:
            method = 'dagostino'
        else:
            method = 'anderson'
    
    if method == 'shapiro':
        if n > 5000:
            print(f"Warning: Shapiro-Wilk test is not reliable for n > 5000 (n={n}). Using D'Agostino test instead.")
            return check_normality(data, alpha, 'dagostino')
        stat, p_value = shapiro(data)
        is_normal = p_value > alpha
        return is_normal, p_value, "Shapiro-Wilk", stat
        
    elif method == 'dagostino':
        stat, p_value = normaltest(data)
        is_normal = p_value > alpha
        return is_normal, p_value, "D'Agostino-Pearson", stat
        
    elif method == 'anderson':
        result = anderson(data, dist='norm')
        stat = result.statistic
        critical_val = result.critical_values[2]
        is_normal = stat < critical_val
        return is_normal, critical_val, "Anderson-Darling", stat
    
    else:
        raise ValueError(f"Unknown method: {method}")


def paired_sample_test(data1, data2, alpha=0.05, check_normality_flag=True):
    """Perform a paired sample test (paired t-test or Wilcoxon signed-rank test) on two related samples, with automatic normality checking and effect size calculation. Uses a conservative approach for small sample sizes if normality is not checked."""
    data1 = np.array(data1)
    data2 = np.array(data2)
    differences = data1 - data2
    n = len(differences)
    
    if n < 2:
        raise ValueError(f"Sample size ({n}) is too small for statistical test. Minimum required: 2")
    
    if n < 30 and not check_normality_flag:
        print(f"Small sample size (n={n} < 30). Using Wilcoxon signed-rank test (conservative approach).")
        stat, p_value = wilcoxon(data1, data2, alternative='two-sided', zero_method='wilcox', correction=True, method='auto')
        w1 = stat
        w2 = n*(n+1)/2 - w1
        min_w = min(w1, w2)
        z_score = (min_w - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
        r = z_score / np.sqrt(n)
        return stat, p_value, "Wilcoxon signed-rank test (small sample, conservative)", r
    
    if check_normality_flag:
        is_normal, p_norm, norm_test, _ = check_normality(differences, alpha)
        
        if is_normal:
            print(f"Data follows normal distribution ({norm_test}, p={p_norm:.4f}). Using paired t-test.")
            t_statistic, p_value = ttest_rel(data1, data2, alternative='two-sided')
            cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0
            return t_statistic, p_value, "Paired t-test", cohens_d
        else:
            print(f"Data does not follow normal distribution ({norm_test}, p={p_norm:.4f}). Using Wilcoxon signed-rank test.")
            stat, p_value = wilcoxon(data1, data2, alternative='two-sided', zero_method='wilcox', correction=True, method='auto')
            w1 = stat
            w2 = n*(n+1)/2 - w1
            min_w = min(w1, w2)
            z_score = (min_w - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
            r = z_score / np.sqrt(n)
            return stat, p_value, "Wilcoxon signed-rank test", r
    else:
        t_statistic, p_value = ttest_rel(data1, data2, alternative='two-sided')
        cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0
        return t_statistic, p_value, "Paired t-test", cohens_d


def unpaired_sample_test(data1, data2, alpha=0.05, check_normality_flag=True, equal_var='auto'):
    """Perform an unpaired sample test (independent t-test or Mann-Whitney U test) on two independent samples, with automatic normality checking and effect size calculation. Uses a conservative approach for small sample sizes if normality is not checked."""
    data1 = np.array(data1)
    data2 = np.array(data2)
    n1, n2 = len(data1), len(data2)
    
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Sample sizes too small (n1={n1}, n2={n2}). Minimum required: 2 per group")
    
    small_sample = (n1 < 30) or (n2 < 30)
    
    if small_sample and not check_normality_flag:
        print(f"Small sample size (n1={n1}, n2={n2}). Using Mann-Whitney U test (conservative approach).")
        stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided', method='auto', use_continuity=True)
        u1 = stat
        u2 = n1 * n2 - u1
        min_u = min(u1, u2)
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z_score = (min_u - mean_u) / std_u if std_u > 0 else 0
        r = z_score / np.sqrt(n1 + n2)
        return stat, p_value, "Mann-Whitney U test (small sample, conservative)", r
    
    if check_normality_flag:
        is_normal1, p_norm1, norm_test1, _ = check_normality(data1, alpha)
        is_normal2, p_norm2, norm_test2, _ = check_normality(data2, alpha)
        
        if is_normal1 and is_normal2:
            print(f"Both datasets follow normal distribution ({norm_test1}: p={p_norm1:.4f}, {norm_test2}: p={p_norm2:.4f}).")
            
            if equal_var == 'auto':
                _, p_levene = levene(data1, data2)
                equal_var_assumed = p_levene > alpha
                print(f"Levene's test for equal variances: p={p_levene:.4f}. {'Assuming' if equal_var_assumed else 'Not assuming'} equal variances.")
            else:
                equal_var_assumed = equal_var
                print(f"User specified: {'Assuming' if equal_var_assumed else 'Not assuming'} equal variances.")
            
            t_statistic, p_value = ttest_ind(data1, data2, alternative='two-sided', equal_var=equal_var_assumed)
            test_name = "Independent t-test" + (" (normal)" if equal_var_assumed else " (Welch's)")
            
            pooled_std = np.sqrt(((n1-1)*np.var(data1, ddof=1) + (n2-1)*np.var(data2, ddof=1)) / (n1+n2-2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
            
            return t_statistic, p_value, test_name, cohens_d
            
        else:
            print(f"At least one dataset does not follow normal distribution ({norm_test1}: p={p_norm1:.4f}, {norm_test2}: p={p_norm2:.4f}). Using Mann-Whitney U test.")
            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided', method='auto', use_continuity=True)
            u1 = stat
            u2 = n1 * n2 - u1
            min_u = min(u1, u2)
            mean_u = n1 * n2 / 2
            std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z_score = (min_u - mean_u) / std_u if std_u > 0 else 0
            r = z_score / np.sqrt(n1 + n2)
            
            return stat, p_value, "Mann-Whitney U test", r
    else:
        t_statistic, p_value = ttest_ind(data1, data2, alternative='two-sided', equal_var=False)
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
        return t_statistic, p_value, "Welch's t-test", cohens_d


def significant_label(p):
    """Return a significance label based on the p-value."""
    if p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def plot_paired_statistics(data, color, ylabel=None, type=None, data_type=None, save_path=None, ylim=None, xticks=None, figure_save_format=None):
    """Plot paired sample statistics for two conditions with significance annotation. Automatically checks normality and selects appropriate test, then visualizes the results with error bars and significance markers."""
    column1 = np.array([row[0] for row in data])
    column2 = np.array([row[1] for row in data])
    mask = (
        (column1 != None) & 
        (column2 != None) & 
        (~np.isnan(column1.astype(float))) & 
        (~np.isnan(column2.astype(float)))
    )
    column1 = column1[mask]
    column2 = column2[mask]
    if ylabel == '%Cell' or ylabel == 'Average %DF/F' or ylabel == '%Ensemble Size' or ylabel == 'Peak %DF/F' or ylabel == 'AUC %DF/F':
        column1 = column1 * 100
        column2 = column2 * 100
    
    column1 = np.asarray(column1).flatten().astype(float)
    column2 = np.asarray(column2).flatten().astype(float)

    stat, p, test_name, effect_size = paired_sample_test(column1, column2)
    df = len(column1) - 1
    column1_mean = np.mean(column1)
    column2_mean = np.mean(column2)
    column1_sem = np.std(column1) / np.sqrt(len(column1))
    column2_sem = np.std(column2) / np.sqrt(len(column2))
    plt.figure(figsize=(3, 6))
    ax = plt.subplot(1, 1, 1)
    plt.plot([1, 2], [column1, column2], "#C8C8CB", linewidth=2)
    plt.plot([1, 2], [column1_mean, column2_mean], color, linewidth=4)
    plt.plot([1, 1], [column1_mean - column1_sem, column1_mean + column1_sem], color, linewidth=4)
    plt.plot([2, 2], [column2_mean - column2_sem, column2_mean + column2_sem], color, linewidth=4)
    plt.plot([1, 2], [1.1*max(max(column1), max(column2)), 1.1*max(max(column1), max(column2))], color='black', linewidth=2.5)
    plt.plot([1, 1], [1.05*max(max(column1), max(column2)), 1.1*max(max(column1), max(column2))], color='black', linewidth=2.5)
    plt.plot([2, 2], [1.05*max(max(column1), max(column2)), 1.1*max(max(column1), max(column2))], color='black', linewidth=2.5)
    plt.text(1.5, 1.2*max(max(column1), max(column2)), f'{significant_label(p)}', ha='center', va='bottom', fontsize=12)
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(ylabel, fontsize=16)
    if xticks is not None:
        plt.xticks([1, 2], xticks)
    else:
        plt.xticks([1, 2], ['Pre', 'Post'])
    plt.yticks(fontsize=14)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.tick_params(width=2.5, length=6)
    plt.xlim(0, 3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if test_name == "Paired t-test":
        t = stat
        cohens_d = effect_size
        print(f"column1(mean={column1_mean:.2f},sem={column1_sem:.2f},n={len(column1)})\ncolumn2(mean={column2_mean:.2f},sem={column2_sem:.2f},n={len(column2)})\n{test_name}(p={p:.3},t={t:.3},df={df},Cohen's d={cohens_d:.3f})")
    elif "Wilcoxon signed-rank test" in test_name:
        w1 = stat
        w2 = len(column1) * (len(column1) + 1) / 2 - w1
        r = effect_size
        print(f"column1(mean={column1_mean:.2f},sem={column1_sem:.2f},n={len(column1)})\ncolumn2(mean={column2_mean:.2f},sem={column2_sem:.2f},n={len(column2)})\n{test_name}(p={p:.3},w1={w1:.3},w2={w2:.3},df={df},r={r:.3f})")

    plt.savefig(rf'{save_path}\{type}_{data_type}_statistic_plot.{figure_save_format}', dpi=300)
    plt.show()


def plot_paired2_statistics(group1, group2, color1, color2, ylabel=None, type=None, data_type=None, save_path=None, ylim=None, xticks=None, figure_save_format=None):
    """Plot paired sample statistics for two groups with significance annotation. Automatically checks normality and selects appropriate test, then visualizes the results with error bars and significance markers."""
    group1_column1 = np.array([row[0] for row in group1])
    group1_column2 = np.array([row[1] for row in group1])
    group2_column1 = np.array([row[0] for row in group2])
    group2_column2 = np.array([row[1] for row in group2])
    mask1 = (
        (group1_column1 != None) & 
        (group1_column2 != None) & 
        (~np.isnan(group1_column1.astype(float))) & 
        (~np.isnan(group1_column2.astype(float)))
    )
    mask2 = (
        (group2_column1 != None) & 
        (group2_column2 != None) & 
        (~np.isnan(group2_column1.astype(float))) & 
        (~np.isnan(group2_column2.astype(float)))
    )
    group1_column1 = group1_column1[mask1]
    group1_column2 = group1_column2[mask1]
    group2_column1 = group2_column1[mask2]
    group2_column2 = group2_column2[mask2]
    group1_column1 = np.asarray(group1_column1).flatten().astype(float)
    group1_column2 = np.asarray(group1_column2).flatten().astype(float)
    group2_column1 = np.asarray(group2_column1).flatten().astype(float)
    group2_column2 = np.asarray(group2_column2).flatten().astype(float)
    if ylabel == '%Cell' or ylabel == 'Average %DF/F' or ylabel == '%Ensemble Size':
        group1_column1 = group1_column1*100
        group1_column2 = group1_column2*100
        group2_column1 = group2_column1*100
        group2_column2 = group2_column2*100
        
    stat1, p1, test_name1, effect1 = paired_sample_test(group1_column1, group1_column2)
    df1 = len(group1_column1) - 1
    stat2, p2, test_name2, effect2 = paired_sample_test(group2_column1, group2_column2)
    df2 = len(group2_column1) - 1
    group1_column1_mean = np.mean(group1_column1)
    group1_column2_mean = np.mean(group1_column2)
    group1_column1_sem = np.std(group1_column1) / np.sqrt(len(group1_column1))
    group1_column2_sem = np.std(group1_column2) / np.sqrt(len(group1_column2))
    group2_column1_mean = np.mean(group2_column1)
    group2_column2_mean = np.mean(group2_column2)
    group2_column1_sem = np.std(group2_column1) / np.sqrt(len(group2_column1))
    group2_column2_sem = np.std(group2_column2) / np.sqrt(len(group2_column2))
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    plt.plot([1, 2], [group1_column1, group1_column2], "#C8C8CB", linewidth=2)
    plt.plot([1, 2], [group1_column1_mean, group1_column2_mean], color1, linewidth=4)
    plt.plot([1, 1], [group1_column1_mean - group1_column1_sem, group1_column1_mean + group1_column1_sem], color1, linewidth=4)
    plt.plot([2, 2], [group1_column2_mean - group1_column2_sem, group1_column2_mean + group1_column2_sem], color1, linewidth=4)
    plt.plot([1, 2], [1.1*max(max(group1_column1), max(group1_column2)), 1.1*max(max(group1_column1), max(group1_column2))], color='black', linewidth=2.5)
    plt.plot([1, 1], [1.05*max(max(group1_column1), max(group1_column2)), 1.1*max(max(group1_column1), max(group1_column2))], color='black', linewidth=2.5)
    plt.plot([2, 2], [1.05*max(max(group1_column1), max(group1_column2)), 1.1*max(max(group1_column1), max(group1_column2))], color='black', linewidth=2.5)
    plt.text(1.5, 1.2*max(max(group1_column1), max(group1_column2)), f'{significant_label(p1)}', ha='center', va='bottom',fontsize=12)
    plt.plot([3, 4], [group2_column1, group2_column2], "#C8C8CB", linewidth=2)
    plt.plot([3, 4], [group2_column1_mean, group2_column2_mean], color2, linewidth=4)
    plt.plot([3, 3], [group2_column1_mean - group2_column1_sem, group2_column1_mean + group2_column1_sem], color2, linewidth=4)
    plt.plot([4, 4], [group2_column2_mean - group2_column2_sem, group2_column2_mean + group2_column2_sem], color2, linewidth=4)
    plt.plot([3, 4], [1.1*max(max(group2_column1), max(group2_column2)), 1.1*max(max(group2_column1), max(group2_column2))], color='black', linewidth=2.5)
    plt.plot([3, 3], [1.05*max(max(group2_column1), max(group2_column2)), 1.1*max(max(group2_column1), max(group2_column2))], color='black', linewidth=2.5)
    plt.plot([4, 4], [1.05*max(max(group2_column1), max(group2_column2)), 1.1*max(max(group2_column1), max(group2_column2))], color='black', linewidth=2.5)
    plt.text(3.5, 1.2*max(max(group2_column1), max(group2_column2)), f'{significant_label(p2)}', ha='center', va='bottom', fontsize=12)
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(ylabel, fontsize=16)
    if xticks is not None:
        plt.xticks([1, 2, 3, 4], xticks)
    else:
        plt.xticks([1, 2, 3, 4], ['Pre', 'Post', 'Pre', 'Post'])
    plt.yticks(fontsize=14)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.tick_params(width=2.5, length=6)
    plt.xlim(0, 5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if test_name1 == "Paired t-test":
        t1 = stat1
        cohens_d1 = effect1
        print(f"Group 1:\ncolumn1(mean={group1_column1_mean:.2f},sem={group1_column1_sem:.2f},n={len(group1_column1)})\ncolumn2(mean={group1_column2_mean:.2f},sem={group1_column2_sem:.2f},n={len(group1_column2)})\n{test_name1}(p={p1:.3},t={t1:.3},df={df1},Cohen's d={cohens_d1:.3f})")
    elif "Wilcoxon signed-rank test" in test_name1:
        w11 = stat1
        w12 = len(group1_column1) * (len(group1_column1) + 1) / 2 - w11
        r1 = effect1
        print(f"Group 1:\ncolumn1(mean={group1_column1_mean:.2f},sem={group1_column1_sem:.2f},n={len(group1_column1)})\ncolumn2(mean={group1_column2_mean:.2f},sem={group1_column2_sem:.2f},n={len(group1_column2)})\n{test_name1}(p={p1:.3},w1={w11:.3},w2={w12:.3},df={df1},r={r1:.3f})")
    if test_name2 == "Paired t-test":
        t2 = stat2
        cohens_d2 = effect2
        print(f"Group 2:\ncolumn1(mean={group2_column1_mean:.2f},sem={group2_column1_sem:.2f},n={len(group2_column1)})\ncolumn2(mean={group2_column2_mean:.2f},sem={group2_column2_sem:.2f},n={len(group2_column2)})\n{test_name2}(p={p2:.3},t={t2:.3},df={df2},Cohen's d={cohens_d2:.3f})")
    elif "Wilcoxon signed-rank test" in test_name2:
        w21 = stat2
        w22 = len(group2_column1) * (len(group2_column1) + 1) / 2 - w21
        r2 = effect2
        print(f"Group 2:\ncolumn1(mean={group2_column1_mean:.2f},sem={group2_column1_sem:.2f},n={len(group2_column1)})\ncolumn2(mean={group2_column2_mean:.2f},sem={group2_column2_sem:.2f},n={len(group2_column2)})\n{test_name2}(p={p2:.3},w1={w21:.3},w2={w22:.3},df={df2},r={r2:.3f})")

    plt.savefig(rf'{save_path}\{type}_{data_type}_paired2_statistic_plot.{figure_save_format}', dpi=300)
    plt.show()


def plot_unpaired_statistics(data1, data2, color1, color2, ylabel=None, type=None, data_type=None, save_path=None, ylim=None, xticks=None, figure_save_format=None):
    """Plot unpaired sample statistics for two groups with significance annotation. Automatically checks normality and selects appropriate test, then visualizes the results with error bars and significance markers."""
    data1 = np.asarray(data1).flatten().astype(float)*100
    data2 = np.asarray(data2).flatten().astype(float)*100
    stat, p, test_name, effect_size = unpaired_sample_test(data1, data2)
    df = len(data1) + len(data2) - 2
    data1_mean = np.mean(data1)
    data2_mean = np.mean(data2)
    data1_sem = np.std(data1) / np.sqrt(len(data1))
    data2_sem = np.std(data2) / np.sqrt(len(data2))
    plt.figure(figsize=(3, 6))
    ax = plt.subplot(1, 1, 1)
    plt.axhline(0, color='black', linewidth=2.5, linestyle='-')
    plt.bar([1, 2], [data1_mean, data2_mean], color=[color1, color2], width=0.6)
    if data1_mean > 0:
        plt.plot([1, 1], [data1_mean, data1_mean + data1_sem], color='black', linewidth=2.5)
        plt.plot([0.85, 1.15], [data1_mean + data1_sem, data1_mean + data1_sem], color='black', linewidth=2.5)
    else:
        plt.plot([1, 1], [data1_mean - data1_sem, data1_mean], color='black', linewidth=2.5)
        plt.plot([0.85, 1.15], [data1_mean - data1_sem, data1_mean - data1_sem], color='black', linewidth=2.5)
        
    if data2_mean > 0:
        plt.plot([2, 2], [data2_mean, data2_mean + data2_sem], color='black', linewidth=2.5)
        plt.plot([1.85, 2.15], [data2_mean + data2_sem, data2_mean + data2_sem], color='black', linewidth=2.5)
    else:
        plt.plot([2, 2], [data2_mean - data2_sem, data2_mean], color='black', linewidth=2.5)
        plt.plot([1.85, 2.15], [data2_mean - data2_sem, data2_mean - data2_sem], color='black', linewidth=2.5)
        
    if data1_mean > 0 or data2_mean > 0:
        plt.plot([1, 2], [1.1*max(data1_mean + data1_sem, data2_mean + data2_sem), 1.1*max(data1_mean + data1_sem, data2_mean + data2_sem)], color='black', linewidth=2.5)
        plt.plot([1, 1], [1.05*max(data1_mean + data1_sem, data2_mean + data2_sem), 1.1*max(data1_mean + data1_sem, data2_mean + data2_sem)], color='black', linewidth=2.5)
        plt.plot([2, 2], [1.05*max(data1_mean + data1_sem, data2_mean + data2_sem), 1.1*max(data1_mean + data1_sem, data2_mean + data2_sem)], color='black', linewidth=2.5)
        plt.text(1.5, 1.2*max(data1_mean + data1_sem, data2_mean + data2_sem), f'{significant_label(p)}', ha='center', va='bottom', fontsize=12)
    else:
        plt.plot([1, 2], [1.1*min(data1_mean - data1_sem, data2_mean - data2_sem), 1.1*min(data1_mean - data1_sem, data2_mean - data2_sem)], color='black', linewidth=2.5)
        plt.plot([1, 1], [1.05*min(data1_mean - data1_sem, data2_mean - data2_sem), 1.1*min(data1_mean - data1_sem, data2_mean - data2_sem)], color='black', linewidth=2.5)
        plt.plot([2, 2], [1.05*min(data1_mean - data1_sem, data2_mean - data2_sem), 1.1*min(data1_mean - data1_sem, data2_mean - data2_sem)], color='black', linewidth=2.5)
        plt.text(1.5, 1.2*min(data1_mean - data1_sem, data2_mean - data2_sem), f'{significant_label(p)}', ha='center', va='top', fontsize=12)
    
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(ylabel, fontsize=16)
    if xticks is not None:
        plt.xticks([1, 2], xticks)
    else:
        plt.xticks([1, 2], ['A', 'B'])
    plt.yticks(fontsize=14)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.tick_params(width=2.5, length=6)
    plt.xlim(0, 3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if "Independent t-test (normal)" in test_name or "Welch's" in test_name:
        t = stat
        cohens_d = effect_size
        print(f"data1(mean={data1_mean:.2f},sem={data1_sem:.2f},n={len(data1)})\ndata2(mean={data2_mean:.2f},sem={data2_sem:.2f},n={len(data2)})\n{test_name}(p={p:.3},t={t:.3},df={df},cohens_d={cohens_d:.3f})")
    elif test_name == "Mann-Whitney U test":
        u1 = stat
        u2 = (len(data1) * len(data2)) - u1
        r = effect_size
        print(f"data1(mean={data1_mean:.2f},sem={data1_sem:.2f},n={len(data1)})\ndata2(mean={data2_mean:.2f},sem={data2_sem:.2f},n={len(data2)})\n{test_name}(p={p:.3},u1={u1:.3},u2={u2:.3},r={r:.3f})")
        
    plt.savefig(rf'{save_path}\{type}_{data_type}_unpaired_statistic_plot.{figure_save_format}', dpi=300)
    plt.show()