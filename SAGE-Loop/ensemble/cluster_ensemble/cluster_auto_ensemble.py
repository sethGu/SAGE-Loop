import os
import sys

# 添加项目根目录到
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # 当前文件所在目录
                 "..", "..")                  # 向上跳两级
)
sys.path.append(project_root)

# from mycaafe import CAAFEClassifier  # Automated Feature Engineering for tabular datasets
from sklearn.cluster import KMeans
from utils.model_generate import build_prompt_samples, get_clustering_model_prompt, generate_model, \
    get_clustering_model_prompt_v2

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pickle
import numpy as np
import random
import argparse
import warnings
import copy
import gc
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ensemble.cluster_ensemble_utils import (
    cluster_ensemble_mcla,
    select_top_models_with_adaptive_threshold,
    cluster_ensemble_stacking_unsupervised,
    stacking_cluster,
    eac_ensemble,
    get_param_prompt_oldSklearn,
    cluster_ensemble_caps,
    cluster_ensemble_mcla_wocluater,
    get_param_prompt)
from utils.utils import format_mean_std, format_mean_std_four

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


# 加载本地数据集
def load_origin_data_2(loc, seed=0):
    # 读取数据集
    with open(loc, 'rb') as f:
        ds = pickle.load(f)

    df_train = ds[1]
    df_test = ds[2]
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train.fillna(0, inplace=True)
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.fillna(0, inplace=True)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    target_column_name = ds[4][-1]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 数据集描述，包含列描述
    dataset_description = ds[-1]
    # print(dataset_description)
    n_clusters = len(np.unique(df[target_column_name]))

    return df, n_clusters, target_column_name, dataset_description

# 加载本地数据集
def load_origin_data(loc):
    with open(loc, 'rb') as f:
        ds = pickle.load(f)
    target_column_name = ds[4][-1]
    df = ds[1]
    dataset_description = ds[-1]
    n_clusters = len(np.unique(df[target_column_name]))
    return df, n_clusters, target_column_name, dataset_description

def base_model(n_clusters, seed):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return kmeans


def to_pd(df, target_name):
    y = df[target_name]
    x = df.drop(target_name, axis=1)

    return x, y


# 执行生成的代码
def code_exec(code):
    try:
        # 尝试编译检查（compile 成 AST 再执行）
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, globals())
        return None
    except Exception as e:
        print("Code could not be executed:", e)
        return str(e)


# 计算统计指标
def print_stats(name, values):
    print(f"{name}: {np.mean(values):.2f} ± {np.std(values):.2f}")


def print_rmsle(name, values):
    print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def clean_llm_code(code: str) -> str:
    import re
    # 去除 ``` 开头的代码块标记和末尾附加内容
    code = re.sub(r"^```python\s*", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"```$", "", code.strip())

    # 清除 <end> 和非代码文字（可能来自 LLM）
    code = re.sub(r"<end>", "", code)

    # 移除 LLM 输出中的解释段或文本开头错误提示
    lines = code.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("class mycluster") or line.strip().startswith("import") or line.strip().startswith(
                "from"):
            cleaned_lines.append(line)
        elif cleaned_lines:  # 如果已开始记录代码块，继续添加后续代码
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def summarize_dataset(df, sample_size=10):
    summary = {
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "feature_types": {},
        "feature_ranges": {},
        "mean_std": {},
        "sample_head": df.head(sample_size).to_dict(orient='records')
    }

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary["feature_types"][col] = "numerical"
            summary["feature_ranges"][col] = [df[col].min(), df[col].max()]
            summary["mean_std"][col] = [df[col].mean(), df[col].std()]
        else:
            summary["feature_types"][col] = "categorical"
            summary["feature_ranges"][col] = list(df[col].unique())[:10]  # limit unique values for readability

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default="0", type=str)
    parser.add_argument('-s', '--default_seed', default=42, type=int)
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str)
    # parser.add_argument('-l', '--llm', default='gpt-4o', type=str, help='LLM')
    parser.add_argument('-e', '--exam_iterations', default=5, type=int)
    parser.add_argument('-f', '--feat_iterations', default=5, type=int)
    parser.add_argument('-m', '--model_iterations', default=5, type=int)
    parser.add_argument('-p', '--param_iterations', default=5, type=int)
    args = parser.parse_args()

    model_tab = 1

    """
    openAI API 设置 
    """
    # TODO 替换为你自己的 API 地址和 Key
    base_url = '' # API 地址
    api_key = '' # API Key

    # for ds_name in ('breast','glass','iris','students','seeds'):
    for ds_name in ['students']:
        print(f"=========== Dataset {ds_name} ===========")
        loc = f"{project_root}/data/" + ds_name + ".pkl"

        ari_list_ensemble = []
        nmi_list_ensemble = []
        ARI_ensemble_MAX = 0
        NMI_ensemble_MAX = 0

        exam_iter = args.exam_iterations

        for exp in range(exam_iter):
            print(f"=========== Experiment {exp + 1}/{exam_iter} ===========")
            test_ari_list = []
            test_nmi_list = []
            seed = args.default_seed + exp
            random.seed(seed)
            np.random.seed(seed)
            df, n_clusters, target_column_name, dataset_description = load_origin_data(loc)
            baseline_model = base_model(n_clusters, seed)

            X_aug, y_true = to_pd(df, target_column_name)
            s = build_prompt_samples(X_aug)

            # model_prompt = get_clustering_model_prompt(
            #     target_column_name=target_column_name,
            #     samples=s,
            #     n_clusters=n_clusters
            # )
            summary = summarize_dataset(df)
            model_prompt = get_clustering_model_prompt_v2(
                samples=s,
                dataset_summary=summary
            )

            model_messages = [
                {
                    "role": "system",
                    "content": "You are a top-level clustering algorithm expert. Your task is to help me iteratively search for the most suitable clustering model that performs best on a clustering task based on the ARI (Adjusted Rand Index) metric. Your answer should only generate code.",
                },
                {
                    "role": "user",
                    "content": model_prompt,
                },
            ]

            best_model_code = None  # 全局最优(多轮)
            best_model_ari = -1
            best_model_nmi = -1
            base_models = []

            # ========== model_iterations 主循环 ==============
            for i in range(args.model_iterations):
                try:
                    code = generate_model(args.llm, model_messages,base_url,api_key)
                    code = clean_llm_code(code)
                except Exception as e:
                    print("Error in LLM API." + str(e))
                    continue

                # 动态修改类名
                new_class_name = f"mycluster_{i + 1}"
                code = code.replace("class mycluster:", f"class {new_class_name}:")

                e = code_exec(code)
                # err = code_exec(param_code)
                print(f"After code_exec: err={e}, class_in_globals={new_class_name in globals()}")

                if e is not None:
                    model_messages += [
                        {"role": "assistant", "content": code},
                        {"role": "user", "content": f"""
                            Code failed: {type(e)} {e}
                            ```python{code}```
                            Fix the error and provide new code:
                            """},
                    ]
                    continue

                try:
                    print('---------------------\n' + code)
                    # 使用新的类名创建模型
                    model_class = globals()[new_class_name]
                    # model = model_class(n_clusters=n_clusters)
                    model = model_class()
                    model_copy = copy.deepcopy(model)
                    model_copy.name = 'myCluster' + str(i + 1)
                    y_pred = model.fit_predict(X_aug)
                except Exception as e:
                    print("Model code execution failed: " + str(e))
                    model_messages += [
                        {"role": "assistant", "content": code},
                        {"role": "user", "content": f"""
                            Execution error: {type(e)} {e}
                            ```python{code}```
                            Provide corrected version:
                            """},
                    ]
                    continue

                test_ari = adjusted_rand_score(y_true, y_pred) * 100
                test_nmi = normalized_mutual_info_score(y_true, y_pred) * 100
                # del model
                # gc.collect()

                # ========== 参数优化仅针对当前模型 ==============
                param_best_code = code
                param_best_ari = test_ari
                param_best_nmi = test_nmi

                final_columns = [x for x in X_aug.columns]
                param_prompt = get_param_prompt(param_best_code, param_best_ari, dataset_description, X_aug,
                                                final_columns)
                param_messages = [
                    {
                        "role": "system",
                        "content": "You are a clustering optimization assistant. "
                                   "Your task is to help me iteratively search for more suitable hyperparameters based on the given clustering model code, "
                                   "so that the model can perform the best on the clustering task based on the ARI (Adjusted Rand Index) metric. Your answer should only generate code.",
                    },
                    {
                        "role": "user",
                        "content": param_prompt,
                    },
                ]

                param_iter = getattr(args, 'param_iterations', 5)

                for p_iter in range(param_iter):
                    print(f"=========== Parameter Optimization {p_iter + 1}/{param_iter} ===========")
                    try:
                        param_code = generate_model(args.llm, param_messages,base_url,api_key)
                        param_code = clean_llm_code(param_code)
                    except Exception as e:
                        print("Error in LLM API (param tuning):", str(e))
                        break

                    # 动态修改类名
                    param_new_class_name = f"mycluster_{i + 1}_param_{p_iter + 1}"
                    param_code = param_code.replace(f"class mycluster_{i + 1}:", f"class {param_new_class_name}:")
                    param_code = param_code.replace(f"class mycluster_{i + 1}_param_{p_iter}:",
                                                    f"class {param_new_class_name}:")

                    err = code_exec(param_code)
                    print(f"After code_exec: err={err}, class_in_globals={param_new_class_name in globals()}")
                    if err is not None:
                        param_messages += [
                            {"role": "assistant", "content": param_code},
                            {"role": "user", "content": f"""
                                Last param-tuned model code failed: {type(err)} {err}
                                ```python{param_code}```
                                Please only output a corrected code block (no explanation), immediately starting with triple backticks (```python).
                                """},
                        ]
                        continue
                    try:
                        print("Parameter-tuned code:\n", param_code)
                        # 使用新的类名创建模型
                        param_model_class = globals()[param_new_class_name]
                        param_model = param_model_class()
                        y_pred_param = param_model.fit_predict(X_aug)
                        param_ari = adjusted_rand_score(y_true, y_pred_param) * 100
                        param_nmi = normalized_mutual_info_score(y_true, y_pred_param) * 100
                    except Exception as e:
                        print("Model param tuning execution failed:", str(e))
                        continue

                    print(f"param_ari = {param_ari:.4f}, param_best_ari = {param_best_ari:.4f}")
                    if param_ari > param_best_ari:
                        print(f"Param tuning improved ARI: {param_best_ari:.2f} → {param_ari:.2f}")
                        param_best_ari = param_ari
                        param_best_code = param_code
                        model_copy = copy.deepcopy(param_model)
                        model_copy.name = 'myCluster' + str(i + 1)
                    # del param_model
                    # gc.collect()

                    param_messages += [
                        {"role": "assistant", "content": param_code},
                        {"role": "user", "content": f"""
                            Code succeeded.
                            Current ARI: {param_ari:.2f}%, Best ARI so far: {param_best_ari:.2f}%.
                            Best code so far:
                            ```python{param_best_code}```
                            Please further optimize hyperparameters only. Output only new Python code.
                            At the end of your code block, output a single line _params = dict(...) containing all parameters you used for mycluster.
                            When using AgglomerativeClustering, please strictly follow scikit-learn's API: Do NOT use the affinity (or metric) parameter if linkage is 'ward'. For scikit-learn >=1.2, replace affinity with metric. Only use valid parameter combinations supported by your sklearn version.
                            """},
                    ]

                print("parm turning over")

                base_models.append(model_copy)

                # 全局最优统计在外层即可
                if param_best_ari > best_model_ari:
                    best_model_ari = param_best_ari
                    best_model_nmi = param_best_nmi
                    best_model_code = param_best_code

                test_ari_list.append(param_best_ari)
                test_nmi_list.append(param_best_nmi)


            print(f"\n=========== Experiment {exp + 1} Results ===========")
            print("ARI", test_ari_list)
            print("NMI", test_nmi_list)

            # ...（后续模型集成、汇总部分略）

            selected_base_models = select_top_models_with_adaptive_threshold(test_ari_list, test_nmi_list, base_models,
                                                                             top_k=5, threshold_ratio=0.5)
            print("selected_base_models size: ", len(selected_base_models))
            print("开始集成了")
            # 如果 selected_base_models 的大小小于2 ，跳过集成
            if len(selected_base_models) < 2:
                print("selected_base_models 模型数量小于2，Skip ensemble")
                continue

            # 使用 MCLA 自动确定cluster方法
            result = cluster_ensemble_caps(
                X=X_aug,
                clusterers=selected_base_models,
                true_labels=y_true,
            )

            ensemble_metrics = result['ensemble_metrics']

            current_ensemble_ari = ensemble_metrics['ensemble_ari']
            current_ensemble_nmi = ensemble_metrics['ensemble_nmi']

            ARI_ensemble_MAX = max(ARI_ensemble_MAX, current_ensemble_ari)
            NMI_ensemble_MAX = max(NMI_ensemble_MAX, current_ensemble_nmi)

            print(f"\n=========== Ensemble Result {exp + 1} ===========")
            print(f"ARI : {current_ensemble_ari:.4f} ")
            print(f"NMI : {current_ensemble_nmi:.4f}")

            ari_list_ensemble.append(round(current_ensemble_ari, 5) * 100)
            nmi_list_ensemble.append(round(current_ensemble_nmi, 5) * 100)

        summary_lines = [
            f"\n=========== Summary for {ds_name} ({args.exam_iterations} runs) ===========",
            f"ARI: {format_mean_std(ari_list_ensemble)}",
            f"NMI: {format_mean_std(nmi_list_ensemble)}",
            f"ARI_ensemble_MAX: {round(ARI_ensemble_MAX, 4)}",
            f"NMI_ensemble_MAX: {round(NMI_ensemble_MAX, 4)}",
            "\n"
        ]

        for line in summary_lines:
            print(line)
        # with open(f"{project_root}/new_ensemble/cluster_ensemble/cluster_res.txt", "a", encoding="utf-8") as f:
        #     for line in summary_lines:
        #         f.write(line + "\n")
        #     f.write('\n------------------------------------------------------------------\n')
