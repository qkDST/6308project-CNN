import os
import sys
import time
import torch
import argparse
from datetime import datetime


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_vdcnn import train_vdcnn
from train_densenet import train_densenet
from utils.config import Config


def setup_environment():


    config = Config()


    print(f"Data strategy: Use the top {config.data_fraction * 100}% of data")
    print(f" 80% Training, 20% Validation")
    print(f"Early Stopping Settings: Patience={config.early_stopping_patience}, delta={config.early_stopping_delta}")


def run_experiment(model_type, dataset_type):
    """运行单个实验"""
    print(f"\n{'=' * 50}")
    print(f"Experiment start: {model_type.upper()} - {dataset_type.upper()}")
    print(f"{'=' * 50}")

    start_time = time.time()

    try:
        # 训练模型
        if model_type == 'vdcnn':
            test_acc = train_vdcnn(dataset_type)
        else:  # densenet
            test_acc = train_densenet(dataset_type)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✓ Experiment finished: {model_type.upper()} - {dataset_type.upper()}")
        print(f"  Acc: {test_acc:.4f}")
        print(f"  Time: {duration:.2f} s  ({duration / 60:.2f} min)")

        return True

    except Exception as e:
        print(f"\n✗ Experiment failed: {model_type.upper()} - {dataset_type.upper()}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_experiments():
    """运行所有实验组合"""
    experiments = [
        ('vdcnn', 'yahoo'),
        ('vdcnn', 'yelp'),
        ('densenet', 'yahoo'),
        ('densenet', 'yelp')
    ]

    successful_experiments = []
    failed_experiments = []

    print("\nBegin to run all experiments...")
    print("Experiment sequence:")
    for i, (model, dataset) in enumerate(experiments, 1):
        print(f"  {i}. {model.upper()} - {dataset.upper()}")

    for model_type, dataset_type in experiments:
        success = run_experiment(model_type, dataset_type)

        if success:
            successful_experiments.append((model_type, dataset_type))
        else:
            failed_experiments.append((model_type, dataset_type))

        print("\n" + "=" * 60)


    print_summary(successful_experiments, failed_experiments)


def run_specific_experiments(models, datasets):

    successful_experiments = []
    failed_experiments = []

    print(f"\nExperiment:")
    print(f"  model: {', '.join(models)}")
    print(f"  dataset: {', '.join(datasets)}")

    for model_type in models:
        for dataset_type in datasets:
            success = run_experiment(model_type, dataset_type)

            if success:
                successful_experiments.append((model_type, dataset_type))
            else:
                failed_experiments.append((model_type, dataset_type))

            print("\n" + "=" * 60)

    # 输出总结
    print_summary(successful_experiments, failed_experiments)


def print_summary(successful, failed):
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if successful:
        print("\nExperiment Success")
        for model, dataset in successful:
            print(f"  - {model.upper()} - {dataset.upper()}")

    if failed:
        print("\nExperiment Failed")
        for model, dataset in failed:
            print(f"  - {model.upper()} - {dataset.upper()}")

    print(f"\n: {len(successful)} Successed, {len(failed)} Failed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文本分类主训练脚本')
    parser.add_argument('--mode', type=str, choices=['all', 'specific'],
                        default='all', help='运行模式: all(所有实验) 或 specific(指定实验)')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['vdcnn', 'densenet'],
                        help='指定要运行的模型 (vdcnn, densenet)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['yahoo', 'yelp'],
                        help='指定要运行的数据集 (yahoo, yelp)')

    args = parser.parse_args()

    # 设置环境
    setup_environment()

    try:
        # 根据模式运行实验
        if args.mode == 'all':
            run_all_experiments()
        else:
            run_specific_experiments(args.models, args.datasets)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.!")
    except Exception as e:
        print(f"\n\nError occured: {e}")
        import traceback
        traceback.print_exc()

    print("\nTraining finished!")


if __name__ == '__main__':
    main()