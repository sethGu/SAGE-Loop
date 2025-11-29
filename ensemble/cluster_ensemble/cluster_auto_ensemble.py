import os
import sys

# 添加项目根目录到
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # 当前文件所在目录
                 "..", "..")                  # 向上跳两级
)
sys.path.append(project_root)

from mycaafe.run_llm_code import run_llm_code
from mycaafe import CAAFEClassifier  # Automated Feature Engineering for tabular datasets
from sklearn.cluster import KMeans
from utils.model_generate import build_prompt_samples, generate_model, get_clustering_model_prompt_v2

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pickle
import numpy as np
import random
import argparse
import warnings
import copy
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ensemble.cluster_ensemble_utils import (
    select_top_models_with_adaptive_threshold,
    cluster_ensemble_caps,
    get_param_prompt
)

from utils.utils import format_mean_std

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


# 生成特征
def generate_feat(
        base_model,
        df,
        dataset_name,
        round_num,
        llm_model='gpt-3.5-turbo',
        iterations=10,
        target_column_name='class',
        dataset_description=None,
        task_type="clustering",
        base_url:str=None,api_key:str=None
):
    if base_url is None or api_key is None:
        raise ValueError("base_url and api_key must be provided.")
    caafe_clf = CAAFEClassifier(base_classifier=base_model,
                                llm_model=llm_model,
                                iterations=iterations)

    caafe_clf.fit_pandas(df,
                         target_column_name=target_column_name,
                         dataset_description=dataset_description,
                         dataset_name=dataset_name,
                         round_num=round_num,
                         task_type=task_type,
                         base_url=base_url,
                         api_key=api_key
                        )

    df_aug = run_llm_code(caafe_clf.code, df, target_column_name)

    final_columns = [x for x in caafe_clf.final_columns if x != target_column_name]

    # 生成特征
    X_aug = df_aug[final_columns]
    # 标签
    y_true = df_aug[target_column_name]

    # kmeans下不标准化ari指标
    y_pred_aug = base_model.fit_predict(X_aug)
    test_ari_aug = adjusted_rand_score(y_true, y_pred_aug) * 100

    # 标准化结果
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_columns)
    # kmeans下标准化ari指标
    y_pred_scaled = base_model.fit_predict(X_scaled_df)
    test_ari_scaled = adjusted_rand_score(y_true, y_pred_scaled) * 100
    # print(f'test_ari_aug:{test_ari_aug}')
    # print(f'test_ari_scaled:{test_ari_scaled}')

    # 选择合适的数据预处理方式：不标准化/标准化
    if test_ari_aug > test_ari_scaled:
        return X_aug, y_true
    else:
        return X_scaled_df, y_true

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default="0", type=str)
    parser.add_argument('-s', '--default_seed', default=42, type=int)
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str)
    parser.add_argument('-e', '--exam_iterations', default=2, type=int)
    parser.add_argument('-f', '--feat_iterations', default=5, type=int)
    parser.add_argument('-m', '--model_iterations', default=4, type=int)
    parser.add_argument('-p', '--param_iterations', default=1, type=int)
    parser.add_argument('-d', '--dataset',default="concrete",help="数据集") # 'breast','glass','iris','students','seeds'
    parser.add_argument("--enable_optimization", action='store_true', default=True, 
                        help="是否启用超参优化逻辑")
    parser.add_argument("--enable_feedback", action='store_true', default=True, 
                        help="是否启用LLM模型生成反馈逻辑")
    args = parser.parse_args()


    """
    openAI API 设置 
    """
    env_url = os.getenv("OPENAI_BASE_URL")
    env_key = os.getenv("OPENAI_API_KEY")
    manual_url = ""
    manual_key = ""

    if env_url and env_key:
        base_url = env_url
        api_key = env_key
    elif manual_url and manual_key:
        base_url = manual_url
        api_key = manual_key
    else:
        raise ValueError(
            "No valid OpenAI API configuration found. "
            "Please set environment variables or manual config."
        )

    
    ds_name=args.dataset
    print(f"=========== Dataset {ds_name} ===========")
    # 打印超参优化开关状态
    print(f"超参优化状态: {'启用' if args.enable_optimization else '禁用'}")
    # 打印模型生成反馈开关状态
    print(f"模型生成反馈状态: {'启用' if args.enable_feedback else '禁用'}")

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

        # 加载数据
        df, n_clusters, target_column_name, dataset_description = load_origin_data(loc)
        baseline_model = base_model(n_clusters, seed)

        # ========== 生成特征 ==========
        X_aug, y_true = generate_feat(
            base_model=baseline_model,
            df=df,
            dataset_name=ds_name,
            round_num=exp + 1,
            llm_model=args.llm,
            iterations=args.feat_iterations,
            target_column_name=target_column_name,
            dataset_description=dataset_description,
            task_type='clustering',
            base_url=base_url,
            api_key=api_key
        )

        # 构建初始样本及模型提示词
        s = build_prompt_samples(X_aug)
        summary = summarize_dataset(df)
        model_prompt = get_clustering_model_prompt_v2(
            samples=s,
            dataset_summary=summary
        )
        model_messages = [
            {
                "role": "system",
                "content": (
                    "You are a top-level clustering algorithm expert. "
                    "You help design clustering models optimized for ARI. "
                    "Your output must contain ONLY python code."
                ),
            },
            {"role": "user", "content": model_prompt},
        ]

        best_model_code = None
        best_model_ari = -1
        best_model_nmi = -1
        base_models = []

        # ========== model_iterations ==========
        for i in range(args.model_iterations):
            try:
                code = generate_model(args.llm, model_messages, base_url, api_key)
                code = clean_llm_code(code)
            except Exception as e:
                print("Error in LLM API: " + str(e))
                continue

            new_class_name = f"mycluster_{i + 1}"
            code = code.replace("class mycluster:", f"class {new_class_name}:")

            err = code_exec(code)
            print(f"[code_exec] err={err}, class_exists={new_class_name in globals()}")

            if err:
                if args.enable_feedback:
                    model_messages += [
                        {"role": "assistant", "content": code},
                        {"role": "user", "content": f"Execution failed: {err}\nPlease fix code."},
                    ]
                continue

            try:
                model_class = globals()[new_class_name]
                model = model_class()
            except Exception as e:
                print("Model init failed: " + str(e))
                continue

            model_copy = copy.deepcopy(model)
            model_copy.name = "myCluster" + str(i + 1)

            try:
                y_pred = model.fit_predict(X_aug)
            except Exception as e:
                print("Model predict failed: " + str(e))
                continue

            test_ari = adjusted_rand_score(y_true, y_pred) * 100
            test_nmi = normalized_mutual_info_score(y_true, y_pred) * 100

            # ===========================
            # 超参优化阶段（可开关）
            # ===========================
            param_best_code = code
            param_best_ari = test_ari
            param_best_nmi = test_nmi

            if args.enable_optimization:
                # 构造参数优化提示词
                final_cols = [x for x in X_aug.columns]
                param_prompt = get_param_prompt(param_best_code, param_best_ari,
                                                dataset_description, X_aug, final_cols)

                param_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a clustering optimization assistant. "
                            "Only optimize hyperparameters. Output python code only."
                        ),
                    },
                    {"role": "user", "content": param_prompt},
                ]

                param_iter = args.param_iterations
                for p_iter in range(param_iter):
                    print(f"----- Param Opt {p_iter + 1}/{param_iter} -----")
                    try:
                        param_code = generate_model(args.llm, param_messages, base_url, api_key)
                        param_code = clean_llm_code(param_code)
                    except Exception as e:
                        print("LLM param error:", e)
                        break

                    param_new_class = f"{new_class_name}_param_{p_iter + 1}"
                    param_code = param_code.replace(new_class_name + ":", param_new_class + ":")

                    err = code_exec(param_code)
                    print(f"[param code_exec] err={err}, exists={param_new_class in globals()}")

                    if err:
                        if args.enable_feedback:
                            param_messages += [
                                {"role": "assistant", "content": param_code},
                                {"role": "user",
                                    "content": f"Param model failed: {err}\nFix and output code."},
                            ]
                        continue

                    try:
                        pc = globals()[param_new_class]
                        param_model = pc()
                        y_pred_param = param_model.fit_predict(X_aug)
                    except Exception as e:
                        print("Param model failed:", e)
                        continue

                    param_ari = adjusted_rand_score(y_true, y_pred_param) * 100
                    param_nmi = normalized_mutual_info_score(y_true, y_pred_param) * 100

                    if param_ari > param_best_ari:
                        param_best_ari = param_ari
                        param_best_code = param_code
                        model_copy = copy.deepcopy(param_model)
                        model_copy.name = "myCluster" + str(i + 1)

                    if args.enable_feedback:
                        param_messages += [
                            {"role": "assistant", "content": param_code},
                            {"role": "user",
                                "content": (
                                    f"Current ARI: {param_ari:.2f}, Best: {param_best_ari:.2f}\n"
                                    "Continue optimizing."
                                )},
                        ]

            # ========== 一轮模型结束 ==========

            base_models.append(model_copy)

            if param_best_ari > best_model_ari:
                best_model_ari = param_best_ari
                best_model_nmi = param_best_nmi
                best_model_code = param_best_code

            test_ari_list.append(param_best_ari)
            test_nmi_list.append(param_best_nmi)

            # 模型反馈逻辑
            if args.enable_feedback:
                model_messages += [
                    {"role": "assistant", "content": param_best_code},
                    {"role": "user",
                        "content": (
                            f"The model achieved ARI={param_best_ari:.2f}. "
                            "Please generate a new improved model architecture."
                        )},
                ]
            else:
                model_messages += [
                    {"role": "assistant", "content": param_best_code},
                    {"role": "user",
                        "content": (
                            "Please generate a new improved model architecture."
                        )},
                ]

        # ========== 单次实验结束 ==========

        print(f"\n=== Experiment {exp + 1} Result ===")
        print("ARI:", test_ari_list)
        print("NMI:", test_nmi_list)

        selected_models = select_top_models_with_adaptive_threshold(
            test_ari_list, test_nmi_list, base_models,
            top_k=5, threshold_ratio=0.5
        )

        if len(selected_models) < 2:
            print("Too few models for ensemble → skip")
            continue

        result = cluster_ensemble_caps(
            X_aug, selected_models, y_true
        )

        ensemble_ari = result["ensemble_metrics"]["ensemble_ari"]
        ensemble_nmi = result["ensemble_metrics"]["ensemble_nmi"]

        ARI_ensemble_MAX = max(ARI_ensemble_MAX, ensemble_ari)
        NMI_ensemble_MAX = max(NMI_ensemble_MAX, ensemble_nmi)

        ari_list_ensemble.append(ensemble_ari * 100)
        nmi_list_ensemble.append(ensemble_nmi * 100)

        print(f"[Ensemble] ARI={ensemble_ari:.4f}, NMI={ensemble_nmi:.4f}")

    print("\n========== Summary ==========")
    print(f"ARI: {format_mean_std(ari_list_ensemble)}")
    print(f"NMI: {format_mean_std(nmi_list_ensemble)}")
    print(f"ARI_MAX: {ARI_ensemble_MAX:.4f}")
    print(f"NMI_MAX: {NMI_ensemble_MAX:.4f}")
