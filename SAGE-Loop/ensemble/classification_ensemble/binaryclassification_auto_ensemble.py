import os
import sys

# 添加项目根目录到
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # 当前文件所在目录
                 "..", "..")                  # 向上跳两级
)
sys.path.append(project_root)



import re

from utils.ensembleUtils import getMetaModel_list
from utils.utils import format_mean_std
from utils.CL_ensemble_utils import stacking_ensemble,get_classification_param_prompt,voting_ensemble
from utils.model_generate import (
    build_prompt_samples,
    get_model_prompt,
    generate_model,
)

import copy
import pandas as pd
import torch

# 第三方库导入
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

import numpy as np
import pickle
import random
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


def get_data_split(ds, seed):
    def get_df(X, y):
        df = pd.DataFrame(
            data=np.concatenate([X, np.expand_dims(y, -1)], -1), columns=ds[4]
        )
        cat_features = ds[3]
        for c in cat_features:
            if len(np.unique(df.iloc[:, c])) > 50:
                cat_features.remove(c)
                continue
            df[df.columns[c]] = df[df.columns[c]].astype("int32")
        return df.infer_objects()

    ds = copy.deepcopy(ds)

    X = ds[1].numpy() if type(ds[1]) == torch.Tensor else ds[1]
    y = ds[2].numpy() if type(ds[2]) == torch.Tensor else ds[2]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    df_train = get_df(X_train, y_train)
    df_test = get_df(X_test, y_test)
    df_train.iloc[:, -1] = df_train.iloc[:, -1].astype("category")
    df_test.iloc[:, -1] = df_test.iloc[:, -1].astype("category")

    return ds, df_train, df_test

def load_origin_data(dataset_name, seed=0):
    # 需要走 .pkl 的旧数据集关键词（子串匹配）
    old_keys = ('credit','cd1','cc1','ld1','cc2','cd2','cf1','balance-scale')
    name_l = dataset_name.lower()
    is_old = any(k in name_l for k in old_keys)

    if is_old:
        loc = f"{project_root}/data/{dataset_name}.pkl"
        with open(loc, 'rb') as f:
            ds = pickle.load(f)

        # credit 系列需要先 split，其它旧数据集直接 ds[1]/ds[2]
        if 'credit' in name_l:
            ds, df_train, df_test = get_data_split(ds, seed=seed)
        else:
            df_train, df_test = ds[1], ds[2]

        target_column_name = ds[4][-1]
        dataset_description = ds[-1]
        return df_train, df_test, target_column_name, dataset_description

def base_model(seed):
    rforest = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # 可重复的随机数据划分
    param_grid = {
        "min_samples_leaf": [0.001, 0.01, 0.05],  # 调整范围
        "max_depth": [5, 10, None]  # 新增深度控制
    }
    gsmodel = GridSearchCV(rforest, param_grid, cv=cv, scoring='f1')

    return gsmodel

def code_exec(code):
    try:
        # 尝试编译检查（compile 成 AST 再执行）
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, globals())
        return None
    except Exception as e:
        print("Code could not be executed:", e)
        return str(e)

def print_stats(name, values):
    print(f"{name}: {np.mean(values):.2f} ± {np.std(values):.2f}")

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
        if line.strip().startswith("class myclassifier") or line.strip().startswith("import") or line.strip().startswith(
                "from"):
            cleaned_lines.append(line)
        elif cleaned_lines:  # 如果已开始记录代码块，继续添加后续代码
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def to_pd(df_train, target_name):
    y = df_train[target_name].astype(int)
    x = df_train.drop(target_name, axis=1)

    return x, y


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default="0", type=str, help='GPU设置')
    parser.add_argument('-s', '--default_seed', default=42, type=int, help='随机种子')
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str, help='大模型')
    # parser.add_argument('-l', '--llm', default='gpt-4o', type=str, help='大模型')
    parser.add_argument('-e', '--exam_iterations', default=5, type=int, help='实验次数')
    # parser.add_argument('-f', '--feat_iterations', default=1, type=int, help='特征迭代次数')
    parser.add_argument('-m', '--model_iterations', default=2, type=int, help='模型迭代次数')
    parser.add_argument('-p', '--param_iterations', default=2, type=int,help='参数调优次数')
    args = parser.parse_args()

    """
    openAI API 设置 
    """
    # TODO 替换为你自己的 API 地址和 Key
    base_url = '' # API 地址
    api_key = '' # API Key

    # 模型标签，全局区分 LLM 生成的模型
    model_tab = 1

    for ds_name in ['ds_credit']:
        # cd1 cc1 ld1  cc2 cd2 cf1 balance-scale ds_credit
        print(f"=========== Dataset {ds_name} ===========")

        # 用于存储每次集成学习的指标的结果
        test_acc_list_ensemble = []
        test_f1_list_ensemble = []
        test_auc_list_ensemble = []
        test_pre_list_ensemble = []
        test_rec_list_ensemble = []

        # 新增：用于存储Voting指标的结果
        test_acc_list_ensemble_voting = []
        test_f1_list_ensemble_voting = []
        test_auc_list_ensemble_voting = []
        test_pre_list_ensemble_voting = []
        test_rec_list_ensemble_voting = []

        # 实验次数
        for exp in range(args.exam_iterations):
            print(f"=========== Experiment {exp + 1}/{args.exam_iterations} ===========")
            # 存储每次实验结果的列表
            test_auc_list = []
            seed = args.default_seed + exp
            # 设置随机种子
            random.seed(seed)
            np.random.seed(seed)
            # 加载数据集、加载模型
            df_train_aug, df_test_aug, target_column_name, dataset_description = load_origin_data(ds_name)
            df_train_aug,df_valid_aug = train_test_split(df_train_aug,test_size=0.25,random_state=seed,stratify=df_train_aug[target_column_name])
            baseline_model = base_model(seed)  # 随机森林 model

            # 生成特征 得到数据增强后的训练和测试集
            # print("特征生成中...")
            # df_train_aug, df_test_aug = generate_feat(
            #     base_classifier=baseline_model,  # 评价生成特征的模型是 随机森林
            #     df_train=df_train_aug,
            #     df_test=df_test_aug,
            #     dataset_name=ds_name,
            #     round_num=exp + 1,
            #     llm_model=args.llm,
            #     iterations=args.feat_iterations,
            #     target_column_name=target_column_name,
            #     dataset_description=dataset_description
            # )
            # print("特征生成完成")

            # 数据转换 得到特征矩阵和 标签（目标） 向量
            train_aug_x, train_aug_y = to_pd(df_train_aug, target_column_name)
            val_aug_x, val_aug_y = to_pd(df_valid_aug, target_column_name)
            test_aug_x, test_aug_y = to_pd(df_test_aug, target_column_name)

            # 构造提示词需要的数据格式
            s = build_prompt_samples(df_train_aug)

            # LLM 生成 分类模型代码的提示词 prompt
            model_prompt = get_model_prompt(
                target_column_name=target_column_name,
                samples=s,
            )

            # LLM 生成 分类器模型 提示词
            model_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a top-level machine learning classification expert.\n"
                        "Your task is to help me iteratively search for the most suitable classifier model.\n"
                        "Your primary goal is to maximize the AUC (Area Under the ROC Curve) on the test set.\n"
                        "You must focus on improving AUC more than any other metric.\n"
                        "Your answer should only generate valid Python code."
                    ),
                },
                {
                    "role": "user",
                    "content": model_prompt,  # 保持生成模型的具体提示结构不变
                },
            ]


            # # 将 messages 写入文件，便于调试和查看
            # file_path = project_root + '/new_ensemble/classification_stacking/prompt.txt'
            # import json

            # with open(file_path, 'a') as f:
            #     # 将消息转为 JSON 字符串，确保可读性且适合存储
            #     f.write(json.dumps(model_messages, ensure_ascii=False) + '\n')

            model_iter = args.model_iterations
            best_auc = 0
            best_code = None
            i = 0

            # 参与集成学习基础模型列表
            base_models = []
            # 模型生成迭代
            while i < model_iter:
                try:
                    # 生成下游模型代码
                    code = generate_model(args.llm, model_messages,base_url,api_key)
                    # todo 加 code_clean 代码
                    code = clean_llm_code(code)

                    # 动态修改类名
                    new_class_name = f"myclassifier_{i + 1}"
                    code = re.sub(r'class\s+myclassifier\w*\s*:', f'class {new_class_name}:', code)
                    print(f"----------------------------原始代码-----------------------")
                    print(code)
                except Exception as e:
                    print("Error in LLM API." + str(e))
                    continue

                e = code_exec(code)
                # 检查编译错误
                if e is not None:  # 生成的代码执行出错 将错误信息反馈给LLM以生成修复后的代码
                    model_messages += [
                        {"role": "assistant", "content": code},
                        {
                            "role": "user",
                            "content": f"""
                                The classifier code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```
                                Remember, your answer should only generate code.
                                Do not include explanations or comments outside the code block.
                                Generate next code block(fixing error?):
                                """,
                        },
                    ]
                    continue


                try:
                    # 模型实例
                    model_class = globals()[new_class_name]
                    model = model_class()
                    model_copy = copy.deepcopy(model)
                    model.fit(train_aug_x, train_aug_y)
                    pred = model.predict(val_aug_x)
                    proba = model.predict_proba(val_aug_x)
                    model_list_append = model_copy

                except Exception as e:
                    print("Model code execution failed with error:" + str(e))
                    model_messages += [
                        {"role": "assistant", "content": code},
                        {
                            "role": "user",
                            "content": f"""
                            Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next code block(fixing error?):
                            """,
                        },
                    ]
                    continue


                test_auc = roc_auc_score(val_aug_y, proba) * 100

                # todo 在这后面加入参数优化
                print(f"----------------参数优化开始，优化代码 {new_class_name}------------------")
                param_best_code = code
                param_best_auc = test_auc
                param_prompt = get_classification_param_prompt(
                    best_code=param_best_code,
                    best_auc=param_best_auc,
                    dataset_description=dataset_description,
                    X_test=val_aug_x,
                    feature_columns=train_aug_x.columns.tolist(),
                    dataset_name=ds_name,
                    max_rows=10
                )

                param_messages = [
                    {
                        "role": "system",
                        "content": "You are a classification optimization assistant.\n"
                                   "Your task is to help me improve the test AUC of the given classifier\n"
                                   "by tuning hyperparameters only. Your answer must contain only executable Python code."
                    },
                    {
                        "role": "user",
                        "content": param_prompt
                    },
                ]

                # 开始参数优化迭代
                for p_iter in range(args.param_iterations):
                    print(f"++++++ 第 {p_iter + 1} 次优化 +++++++")
                    try:
                        param_code = generate_model(args.llm, param_messages,base_url,api_key)
                        param_code = clean_llm_code(param_code)
                        param_new_class_name = f"myclassifier_{i+1}_param_{p_iter + 1}"
                        param_code = param_code.replace(f"class myclassifier_{i+1}:", f"class {param_new_class_name}:")
                        param_code = param_code.replace(f"class myclassifier_{i+1}_param_{p_iter}:",f"class {param_new_class_name}:")

                        param_err = code_exec(param_code)

                        if param_err is not None:
                            param_messages += [
                                {"role": "assistant", "content": param_code},
                                {"role": "user", "content": "Code failed. Fix it and regenerate."},
                            ]
                            continue

                        print('---------------优化后的代码\n' + param_code)
                        # 使用新的类名创建模型
                        model_class = globals()[param_new_class_name]
                        myclassifier_tuned = model_class()
                        model_copy = copy.deepcopy(myclassifier_tuned)
                        myclassifier_tuned.fit(train_aug_x, train_aug_y)
                        proba_tuned = myclassifier_tuned.predict_proba(val_aug_x)
                        auc_tuned = roc_auc_score(val_aug_y, proba_tuned) * 100

                        if auc_tuned > param_best_auc:
                            print(f"参数优化效果提升：{param_best_auc} --> {auc_tuned}")
                            param_best_auc = auc_tuned
                            param_best_code = param_code
                            model_list_append = model_copy


                        param_messages += [
                            {"role": "assistant", "content": param_code},
                            {"role": "user",
                             "content": f"Current AUC: {auc_tuned:.2f}, Best AUC: {param_best_auc:.2f}. Please improve further."},
                        ]

                    except Exception as e:
                        print("Tuning failed:", str(e))
                        continue

                # 更新全局最佳模型和全局最佳参数
                if best_auc < param_best_auc:
                    best_auc = param_best_auc
                    best_code = param_best_code

                # 加入调参后模型
                base_models.append(model_list_append)

                # 存储结果
                test_auc_list.append(param_best_auc)

                # 打印当前实验详细结果
                print(f"当前实验结果第 {i+1}/{args.model_iterations}")
                print(f"Test  AUC: {param_best_auc:.2f}")
                # while 循环继续
                i = i + 1

                # 下一轮模型生成提示词拼接
                if len(code) > 10:
                    model_messages += [
                        {"role": "assistant", "content": param_best_code},
                        {
                            "role": "user",
                            "content": f"""
                            ✅ The classifier code executed successfully.

                            📈 Current model AUC: {param_best_auc:.4f}
                            🏆 Best historical AUC so far: {best_auc:.4f}

                            Please now propose a new classifier that is **more likely to improve the AUC** on the given test data.
                            The model must differ from all previous ones **by model type or internal structure**.

                            ⚠️ Remember:
                            - You must only output valid Python code for a complete classifier named `myclassifier`.
                            - The class must include all imports and implement: `fit`, `predict`, and `predict_proba`.
                            - Do not repeat models you've already used.
                            - Prioritize models that provide reliable probabilistic outputs to help improve AUC.

                            🎯 Next code block:
                            """,
                        },
                    ]


            """
            集成学习  调用  Stacking and Voting 方法
            """
            # 想要测试的元模型名字列表
            metaModelName_list = [
                # 'RandomForestClassifier',
                # 'XGBClassifier',
                # 'LGBMClassifier',
                # 'CatBoostClassifier',
                # 'SVC',
                # 'DecisionTreeClassifier',
                'LogisticRegression',
                # 'BaggingClassifier',
            ]
            # 想要测试的元模型列表
            metaModelName_list = getMetaModel_list(metaModelName_list)

            for meta_model in metaModelName_list:
                # 调用集成学习方法
                result = stacking_ensemble(base_models, meta_model,train_aug_x,train_aug_y,test_aug_x,test_aug_y)
                stacking_metrics = result['stacking_metrics']

                print(f"\n=========== 第{exp + 1}次Stacking集成结果--{type(meta_model).__name__} ===========")
                print(f"Accuracy : {stacking_metrics['accuracy']:.4f}")
                print(f"F1 Score : {stacking_metrics['f1']:.4f}")
                print(f"AUC      : {stacking_metrics['auc']:.4f}")
                print(f"Precision: {stacking_metrics['precision']:.4f}")
                print(f"Recall   : {stacking_metrics['recall']:.4f}")
                # 保存集成结果
                test_acc_list_ensemble.append(round(stacking_metrics['accuracy'], 5)*100)
                test_f1_list_ensemble.append(round(stacking_metrics['f1'], 5)*100)
                test_pre_list_ensemble.append(round(stacking_metrics['precision'], 5)*100)
                test_rec_list_ensemble.append(round(stacking_metrics['recall'], 5)*100)
                test_auc_list_ensemble.append(round(stacking_metrics['auc'], 5)*100)

                
            """
            新增:Voting 集成
            直接对与本轮生成/调参后的 base_models 进行投票融合，并评估指标
            """
            voting_result = voting_ensemble(base_models, test_aug_x, test_aug_y)
            voting_metrics = voting_result['metrics']

            print(f"\n=========== 第{exp + 1}次 Voting 集成结果 ===========")
            print(f"Accuracy : {voting_metrics['accuracy']:.4f}")
            print(f"F1 Score : {voting_metrics['f1']:.4f}")
            print(f"AUC      : {voting_metrics['auc']:.4f}")
            print(f"Precision: {voting_metrics['precision']:.4f}")
            print(f"Recall   : {voting_metrics['recall']:.4f}")

            # 保存Voting集成结果
            test_acc_list_ensemble_voting.append(round(voting_metrics['accuracy'], 5)*100)
            test_f1_list_ensemble_voting.append(round(voting_metrics['f1'], 5)*100)
            test_pre_list_ensemble_voting.append(round(voting_metrics['precision'], 5)*100)
            test_rec_list_ensemble_voting.append(round(voting_metrics['recall'], 5)*100)
            test_auc_list_ensemble_voting.append(round(voting_metrics['auc'], 5)*100)

        # 实验迭代结束，统计集成学习的结果
        print(f"\n=========== 实验结束，共{args.exam_iterations}次集成学习统计结果信息 ===========")
        print("\nStacking 集成学习结果:")
        print('acc: ' + format_mean_std(test_acc_list_ensemble))
        print('f1: ' + format_mean_std(test_f1_list_ensemble))
        print('auc: ' + format_mean_std(test_auc_list_ensemble))
        print('pre: ' + format_mean_std(test_pre_list_ensemble))
        print('rec: ' + format_mean_std(test_rec_list_ensemble))
        print("\nVoting 集成学习结果:")
        print('voting_acc: ' + format_mean_std(test_acc_list_ensemble_voting))
        print('voting_f1: ' + format_mean_std(test_f1_list_ensemble_voting))
        print('voting_auc: ' + format_mean_std(test_auc_list_ensemble_voting))
        print('voting_pre: ' + format_mean_std(test_pre_list_ensemble_voting))
        print('voting_rec: ' + format_mean_std(test_rec_list_ensemble_voting))