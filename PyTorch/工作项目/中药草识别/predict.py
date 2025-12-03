import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime

TEST_DIR = 'test'  # 测试集目录
MODEL_PATH = 'best_model.pth'  # 模型路径
CLASS_MAPPING_PATH = 'class_mapping.json'  # 类别映射文件路径
BATCH_SIZE = 32
SAVE_RESULTS = True  # 是否保存详细结果


# 定义ResNet模型（与训练时保持一致）
class HerbResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(HerbResNet, self).__init__()
        self.backbone = models.resnet50(weights=None)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class TestEvaluator:
    def __init__(self):
        """初始化测试评估器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("中药草识别测试集自动评估")
        print("=" * 60)
        print(f'使用设备: {self.device}')

        # 检查必要文件
        self._check_files()

        # 加载模型
        print('加载模型...')
        self._load_model()

        # 加载类别映射
        with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f'模型加载完成')
        print(f'支持的类别: {[self.class_mapping.get(name, name) for name in self.class_names]}')

    def _check_files(self):
        """检查必要文件是否存在"""
        if not os.path.exists(TEST_DIR):
            raise FileNotFoundError(f"错误: 测试集目录不存在: {TEST_DIR}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"错误: 模型文件不存在: {MODEL_PATH}")

        if not os.path.exists(CLASS_MAPPING_PATH):
            raise FileNotFoundError(f"错误: 类别映射文件不存在: {CLASS_MAPPING_PATH}")

        print("必要文件检查通过")

    def _load_model(self):
        """加载训练好的模型"""
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.class_names = checkpoint['class_names']

        self.model = HerbResNet(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def collect_test_data(self):
        """收集测试集中的所有图片"""
        print(f'\n扫描测试集: {TEST_DIR}')

        all_images = []
        true_labels = []

        # 遍历每个类别文件夹
        for class_name in os.listdir(TEST_DIR):
            class_path = os.path.join(TEST_DIR, class_name)
            if os.path.isdir(class_path):
                class_images = []
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        all_images.append(img_path)
                        true_labels.append(class_name)
                        class_images.append(img_path)

                chinese_name = self.class_mapping.get(class_name, class_name)
                print(f"   {chinese_name}({class_name}): {len(class_images)} 张图片")

        if not all_images:
            raise ValueError("错误: 测试集中没有找到图片")

        print(f"总计: {len(all_images)} 张测试图片")

        return all_images, true_labels

    def predict_images(self, image_paths):
        """对图片列表进行预测"""
        print(f'\n开始预测 {len(image_paths)} 张图片...')

        predicted_labels = []
        confidences = []

        for i, img_path in enumerate(image_paths):
            try:
                # 加载和预处理图片
                image = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)

                # 预测
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)

                predicted_class = self.class_names[predicted_idx.item()]
                predicted_labels.append(predicted_class)
                confidences.append(confidence.item())

            except Exception as e:
                print(f"   警告: 预测失败: {img_path} - {str(e)}")
                predicted_labels.append('unknown')
                confidences.append(0.0)

            # 显示进度
            if (i + 1) % 20 == 0 or (i + 1) == len(image_paths):
                progress = (i + 1) / len(image_paths) * 100
                print(f"   进度: {i + 1}/{len(image_paths)} ({progress:.1f}%)")

        print("预测完成!")
        return predicted_labels, confidences

    def calculate_metrics(self, true_labels, predicted_labels, confidences):
        """计算详细的评估指标"""
        print(f'\n计算评估指标...')

        # 整体准确率
        overall_accuracy = accuracy_score(true_labels, predicted_labels)

        # 混淆矩阵
        unique_classes = sorted(list(set(true_labels)))
        cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)

        # 各类别指标
        class_metrics = {}
        for i, class_name in enumerate(unique_classes):
            # 该类别的样本索引
            class_indices = [j for j, label in enumerate(true_labels) if label == class_name]

            if class_indices:
                class_true = [true_labels[j] for j in class_indices]
                class_pred = [predicted_labels[j] for j in class_indices]
                class_conf = [confidences[j] for j in class_indices]

                # 计算该类别的准确率
                class_accuracy = accuracy_score(class_true, class_pred)

                # 计算平均置信度
                avg_confidence = np.mean(class_conf)

                # 真阳性、假阳性、假阴性
                tp = sum(1 for t, p in zip(class_true, class_pred) if t == class_name and p == class_name)
                fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != class_name and p == class_name)
                fn = sum(1 for t, p in zip(class_true, class_pred) if t == class_name and p != class_name)

                # 精确率和召回率
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                class_metrics[class_name] = {
                    'accuracy': class_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'avg_confidence': avg_confidence,
                    'support': len(class_indices),
                    'correct_predictions': tp
                }

        return {
            'overall_accuracy': overall_accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': cm,
            'unique_classes': unique_classes
        }

    def print_results(self, results):
        """打印详细的评估结果"""
        print('\n' + '=' * 80)
        print('测试集评估结果')
        print('=' * 80)

        # 整体准确率
        overall_acc = results['overall_accuracy']
        print(f'整体准确率: {overall_acc:.4f} ({overall_acc * 100:.2f}%)')

        # 各类别详细指标
        print(f'\n各类别详细指标:')
        print('-' * 100)
        print(
            f'{"类别":<15} {"准确率":<8} {"精确率":<8} {"召回率":<8} {"F1分数":<8} {"平均置信度":<10} {"样本数":<6} {"正确数":<6}')
        print('-' * 100)

        for class_name, metrics in results['class_metrics'].items():
            chinese_name = self.class_mapping.get(class_name, class_name)
            display_name = f"{chinese_name}({class_name})"

            print(f'{display_name:<15} '
                  f'{metrics["accuracy"]:.4f}   '
                  f'{metrics["precision"]:.4f}   '
                  f'{metrics["recall"]:.4f}   '
                  f'{metrics["f1_score"]:.4f}   '
                  f'{metrics["avg_confidence"]:.4f}     '
                  f'{metrics["support"]:<6d} '
                  f'{metrics["correct_predictions"]:<6d}')

        # sklearn分类报告
        print(f'\n详细分类报告:')
        true_labels_for_report = []
        pred_labels_for_report = []

        # 重建标签用于sklearn报告
        for class_name in results['unique_classes']:
            class_metrics = results['class_metrics'][class_name]
            true_labels_for_report.extend([class_name] * class_metrics['support'])
            correct_count = class_metrics['correct_predictions']
            incorrect_count = class_metrics['support'] - correct_count
            pred_labels_for_report.extend([class_name] * correct_count)
            # 为错误预测添加其他类别（简化处理）
            if incorrect_count > 0:
                other_classes = [c for c in results['unique_classes'] if c != class_name]
                pred_labels_for_report.extend(other_classes[:incorrect_count])

        target_names = [self.class_mapping.get(name, name) for name in results['unique_classes']]

        # 使用原始的true_labels和predicted_labels（从evaluate方法传入）
        if hasattr(self, '_last_true_labels') and hasattr(self, '_last_pred_labels'):
            report = classification_report(
                self._last_true_labels,
                self._last_pred_labels,
                target_names=target_names,
                labels=results['unique_classes'],
                digits=4
            )
            print(report)

    def plot_confusion_matrix(self, results):
        """绘制混淆矩阵"""
        print(f'\n绘制混淆矩阵...')

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        cm = results['confusion_matrix']
        class_names_cn = [self.class_mapping.get(name, name) for name in results['unique_classes']]

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=class_names_cn,
                    yticklabels=class_names_cn,
                    cbar_kws={'label': '样本数量'})

        plt.title('测试集混淆矩阵', fontsize=16, pad=20)
        plt.xlabel('预测类别', fontsize=14)
        plt.ylabel('真实类别', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = 'test_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f'混淆矩阵已保存: {save_path}')
        return save_path

    def save_results(self, results, true_labels, predicted_labels, confidences):
        """保存详细结果到文件"""
        if not SAVE_RESULTS:
            return

        print(f'\n保存详细结果...')

        # 准备保存数据
        save_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_directory': TEST_DIR,
            'model_path': MODEL_PATH,
            'overall_accuracy': float(results['overall_accuracy']),
            'total_samples': len(true_labels),
            'class_metrics': {},
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'class_names': results['unique_classes'],
            'class_mapping': self.class_mapping
        }

        # 转换类别指标
        for class_name, metrics in results['class_metrics'].items():
            save_data['class_metrics'][class_name] = {
                'chinese_name': self.class_mapping.get(class_name, class_name),
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'avg_confidence': float(metrics['avg_confidence']),
                'support': int(metrics['support']),
                'correct_predictions': int(metrics['correct_predictions'])
            }

        # 保存到JSON文件
        results_file = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f'详细结果已保存: {results_file}')

    def run_evaluation(self):
        """运行完整的评估流程"""
        try:
            # 1. 收集测试数据
            image_paths, true_labels = self.collect_test_data()

            # 2. 进行预测
            predicted_labels, confidences = self.predict_images(image_paths)

            # 3. 计算指标
            results = self.calculate_metrics(true_labels, predicted_labels, confidences)

            # 保存用于sklearn报告的标签
            self._last_true_labels = true_labels
            self._last_pred_labels = predicted_labels

            # 4. 打印结果
            self.print_results(results)

            # 5. 绘制混淆矩阵
            self.plot_confusion_matrix(results)

            # 6. 保存结果
            self.save_results(results, true_labels, predicted_labels, confidences)

            print('\n' + '=' * 80)
            print('测试集评估完成!')
            print(f'完成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            print('=' * 80)

            return results

        except Exception as e:
            print(f'\n错误: 评估过程中出现错误: {str(e)}')
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    evaluator = TestEvaluator()
    results = evaluator.run_evaluation()

    if results:
        overall_acc = results['overall_accuracy']
        print(f'\n最终结果: 测试集准确率 {overall_acc:.4f} ({overall_acc * 100:.2f}%)')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Qt5Agg")
    main()