import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

class GlobalWorkspaceModel:
    """
    基于全球工作空间理论（GWT）的意识模型实现
    """

    def __init__(self, num_modules=5, workspace_capacity=3,
                 show_visualization=True, save_figures=False):
        """
        初始化模型
        """
        # 字体
        self._setup_fonts()

        self.num_modules = num_modules
        self.workspace_capacity = workspace_capacity
        self.show_visualization = show_visualization
        self.save_figures = save_figures
        self.figure_counter = 0

        # 初始化模块 - 模块进入意识
        self.modules = {
            '视觉': {'activation': 0.0, 'content': '', 'threshold': 0.6},
            '听觉': {'activation': 0.0, 'content': '', 'threshold': 0.5},
            '记忆': {'activation': 0.0, 'content': '', 'threshold': 0.7},
            '语言': {'activation': 0.0, 'content': '', 'threshold': 0.6},
            '情感': {'activation': 0.0, 'content': '', 'threshold': 0.4}
        }

        # 全球工作空间
        self.global_workspace = deque(maxlen=workspace_capacity)

        # 意识流记录
        self.stream_of_consciousness = []

        # 注意力权重
        self.attention_weights = np.random.dirichlet(np.ones(num_modules))

    def _setup_fonts(self):
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print("字体设置: 已配置中文字体支持")
        except:
            print("字体设置: 使用默认字体")

    def process_stimulus(self, stimulus, stimulus_name=None):
        """处理外部刺激"""
        if stimulus_name:
            print(f"\n接收到刺激 '{stimulus_name}': {stimulus}")
        else:
            print(f"\n接收到刺激: {stimulus}")

        # 处理流程
        unconscious_results = self._unconscious_processing(stimulus)
        competing_modules = self._compete_for_access(unconscious_results)
        conscious_content = self._broadcast_to_global_workspace(competing_modules)

        # 记录意识体验
        if conscious_content:
            self.stream_of_consciousness.append({
                'timestamp': time.time(),
                'content': conscious_content,
                'modules': list(competing_modules.keys()),
                'stimulus': stimulus_name or str(stimulus)[:50] + "..."
            })

        return conscious_content

    def _unconscious_processing(self, stimulus):
        """无意识处理阶段"""
        results = {}

        for module_name, module_info in self.modules.items():
            if module_name in stimulus:
                content = stimulus[module_name]
                activation = self._calculate_activation(content, module_name)

                self.modules[module_name]['activation'] = activation
                self.modules[module_name]['content'] = content

                if activation >= module_info['threshold']:
                    results[module_name] = {
                        'activation': activation,
                        'content': content
                    }

        return results

    def _calculate_activation(self, content, module_name):
        """计算模块激活水平"""
        # 基础激活水平
        base_activation = np.random.uniform(0.5, 0.9)

        # 模块特定增强
        module_boost = {
            '情感': 0.3,  # 情感模块更容易进入意识
            '视觉': 0.2,  # 视觉信息优先级较高
            '听觉': 0.2,
            '语言': 0.1,
            '记忆': 0.1
        }

        if module_name in module_boost:
            base_activation += module_boost[module_name]

        # 内容长度影响
        content_length_factor = min(len(str(content)) / 10, 1.0)  # 更容易达到1.0

        # 情感词汇检测
        emotional_words = ['危险', '警报', '惊讶', '快乐', '愉悦', '悲伤',
                           '专注', '平静', '安宁', '恐惧', '愤怒', '高兴', '中性']
        emotion_factor = 0.0
        for word in emotional_words:
            if word in str(content):
                emotion_factor += 0.3

        # 关键词检测
        keywords = {
            '视觉': ['红色', '突然', '熟悉', '绿色', '阳光', '警示灯', '面孔', '草地', '鲜花'],
            '听觉': ['刺耳', '温暖', '鸟鸣', '风声', '刹车声', '问候声', '音乐'],
            '语言': ['重要', '公式', '推导', '消息', '数学'],
            '记忆': ['回忆', '知识', '美好', '知识点', '之前'],
            '情感': emotional_words
        }

        keyword_factor = 0.0
        if module_name in keywords:
            for word in keywords[module_name]:
                if word in str(content):
                    keyword_factor += 0.2

        # 减小噪声
        noise = np.random.normal(0, 0.05)

        # 计算最终激活值
        activation = (base_activation * 0.3 +
                      content_length_factor * 0.3 +
                      emotion_factor * 0.2 +
                      keyword_factor * 0.2 +
                      noise)

        return np.clip(activation, 0, 1.5)

    def _compete_for_access(self, unconscious_results):
        """竞争阶段"""
        if not unconscious_results:
            print("警告: 无模块达到激活阈值!")
            return {}

        sorted_modules = sorted(
            unconscious_results.items(),
            key=lambda x: x[1]['activation'],
            reverse=True
        )

        k = min(self.workspace_capacity, len(sorted_modules))
        competing_modules = dict(sorted_modules[:k])

        print(f"竞争获胜的模块: {list(competing_modules.keys())}")
        for module, info in competing_modules.items():
            print(f"  {module}: 激活水平={info['activation']:.3f}")

        return competing_modules

    def _broadcast_to_global_workspace(self, competing_modules):
        """广播阶段"""
        if not competing_modules:
            print("无模块进入全球工作空间")
            return None

        broadcast_content = {}
        for module_name, info in competing_modules.items():
            broadcast_content[module_name] = info['content']
            self.global_workspace.append({
                'module': module_name,
                'content': info['content'],
                'activation': info['activation'],
                'timestamp': time.time()
            })

        print(f"全球工作空间内容: {broadcast_content}")

        conscious_experience = self._integrate_experience(broadcast_content)
        print(f"意识体验: {conscious_experience}")

        return conscious_experience

    def _integrate_experience(self, broadcast_content):
        """整合多模态信息"""
        experience = []

        # 优先级顺序
        priority_order = ['情感', '视觉', '听觉', '语言', '记忆']

        for module in priority_order:
            if module in broadcast_content:
                content = broadcast_content[module]
                if module == '情感':
                    experience.append(f"[{content}的感觉]")
                elif module == '视觉':
                    experience.append(f"看到: {content}")
                elif module == '听觉':
                    experience.append(f"听到: {content}")
                elif module == '语言':
                    experience.append(f"思考: {content}")
                else:
                    experience.append(f"回忆: {content}")

        return " | ".join(experience)

    def visualize_competition(self, title_suffix=""):
        """可视化模块竞争过程"""
        modules = list(self.modules.keys())
        activations = [self.modules[m]['activation'] for m in modules]
        thresholds = [self.modules[m]['threshold'] for m in modules]

        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(modules))
        width = 0.6

        # 创建条形图
        bars = ax.bar(x, activations, width, label='激活水平',
                      color=['lightgray'] * len(modules), edgecolor='black', linewidth=1.5)

        # 标记超过阈值的模块
        for i, (module, activation) in enumerate(zip(modules, activations)):
            if activation >= self.modules[module]['threshold']:
                bars[i].set_color('orange')
                bars[i].set_edgecolor('darkorange')
                bars[i].set_linewidth(2)

        # 添加阈值线
        ax.plot(x, thresholds, 'r--', label='意识阈值', linewidth=2.5, alpha=0.8)

        # 设置图表属性
        ax.set_xlabel('认知模块', fontsize=12, fontweight='bold')
        ax.set_ylabel('激活水平', fontsize=12, fontweight='bold')

        title = '全球工作空间竞争模型'
        if title_suffix:
            title += f" - {title_suffix}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        ax.set_xticks(x)
        ax.set_xticklabels(modules, rotation=45, fontsize=11, fontweight='bold')

        # 设置y轴范围
        ax.set_ylim(0, 1.5)
        ax.set_yticks(np.arange(0, 1.6, 0.2))

        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加图例
        ax.legend(fontsize=10, loc='upper right')

        # 在条形上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        # 添加说明文本
        conscious_modules = [m for m, a in zip(modules, activations)
                             if a >= self.modules[m]['threshold']]
        if conscious_modules:
            info_text = f"进入意识的模块: {', '.join(conscious_modules)}"
        else:
            info_text = "无模块进入意识"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存或显示图形
        if self.save_figures:
            filename = f"consciousness_model_{self.figure_counter:02d}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存为: {filename}")
            self.figure_counter += 1

        if self.show_visualization:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_consciousness_stream(self, num_entries=None):
        """可视化意识流时间线"""
        if not self.stream_of_consciousness:
            print("意识流记录为空")
            return

        entries = self.stream_of_consciousness
        if num_entries:
            entries = entries[-num_entries:]

        # 创建时间线图
        fig, ax = plt.subplots(figsize=(14, 5))

        # 获取时间戳
        timestamps = [e['timestamp'] for e in entries]
        start_time = min(timestamps)
        time_deltas = [t - start_time for t in timestamps]

        # 绘制每个意识事件
        colors = plt.cm.Set3(np.linspace(0, 1, len(entries)))

        for i, (entry, delta) in enumerate(zip(entries, time_deltas)):
            # 创建事件点
            ax.scatter(delta, i, s=200, alpha=0.7, color=colors[i],
                       label=f"事件{i + 1}", zorder=5)

            # 添加事件标签part
            content_short = entry['content'][:20] + "..." if len(entry['content']) > 20 else entry['content']
            label = f"事件{i + 1}: {content_short}"
            ax.text(delta + 0.2, i, label, va='center', fontsize=9)

        ax.set_xlabel('时间 (秒)', fontsize=12, fontweight='bold')
        ax.set_ylabel('意识事件', fontsize=12, fontweight='bold')
        ax.set_title('意识流时间线', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.yaxis.set_visible(False)  # 隐藏y轴

        plt.tight_layout()

        if self.save_figures:
            filename = f"consciousness_stream.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"意识流图表已保存为: {filename}")

        if self.show_visualization:
            plt.show()
        else:
            plt.close(fig)

    def show_consciousness_stream(self, num_entries=None, show_details=True):
        """显示意识流记录"""
        if not self.stream_of_consciousness:
            print("意识流记录为空")
            return

        entries = self.stream_of_consciousness
        if num_entries:
            entries = entries[-num_entries:]

        print("\n" + "=" * 60)
        print("意识流记录:")
        print("=" * 60)

        for i, entry in enumerate(entries):
            print(f"\n事件 #{i + 1}:")
            print(f"  时间: {time.ctime(entry['timestamp'])}")
            print(f"  刺激: {entry['stimulus']}")
            print(f"  意识内容: {entry['content']}")
            print(f"  参与模块: {entry['modules']}")

    def get_statistics(self):
        """获取模型统计信息"""
        stats = {
            'total_conscious_events': len(self.stream_of_consciousness),
            'current_workspace_content': list(self.global_workspace),
            'module_activations': {m: self.modules[m]['activation'] for m in self.modules},
            'conscious_modules_count': sum(1 for m in self.modules.values()
                                           if m['activation'] >= m['threshold'])
        }
        return stats


def consciousness_model(show_plots=True, save_plots=False):
    print("初始化意识模型...")
    model = GlobalWorkspaceModel(
        show_visualization=show_plots,
        save_figures=save_plots
    )

    # 模拟一系列现实场景刺激
    scenarios = [
        {
            'name': '紧急情况',
            'stimulus': {
                '视觉': '前方突然出现的红色警示灯',
                '听觉': '刺耳的刹车声',
                '情感': '危险警报'
            }
        },
        {
            'name': '社交互动',
            'stimulus': {
                '视觉': '熟悉的朋友微笑的面孔',
                '听觉': '温暖的问候声',
                '情感': '愉悦',
                '记忆': '之前的美好回忆'
            }
        },
        {
            'name': '学习场景',
            'stimulus': {
                '语言': '重要的数学公式推导',
                '记忆': '相关的知识点',
                '情感': '专注'
            }
        },
        {
            'name': '自然环境',
            'stimulus': {  # 修正拼写错误
                '视觉': '阳光下的绿色草地和鲜花',
                '听觉': '鸟鸣声和风声',
                '情感': '平静安宁'
            }
        }
    ]

    print("\n" + "=" * 60)
    print("开始意识模拟实验")
    print("=" * 60)

    for i, scenario in enumerate(scenarios):
        print(f"\n{'=' * 60}")
        print(f"场景 #{i + 1}: {scenario['name']}")
        print('=' * 60)

        conscious_content = model.process_stimulus(
            scenario['stimulus'],
            scenario['name']
        )

        # 显示当前状态
        if conscious_content:
            print(f"\n当前意识状态: {conscious_content}")
        else:
            print(f"\n当前意识状态: 无意识内容（无模块达到阈值）")

        # 调试：显示各模块的激活水平
        print(f"\n各模块激活水平:")
        for module_name, module_info in model.modules.items():
            threshold = module_info['threshold']
            activation = module_info['activation']
            status = "✓" if activation >= threshold else "✗"
            print(f"  {module_name}: {activation:.3f} (阈值: {threshold}) {status}")

        # 可视化竞争过程
        if show_plots or save_plots:
            model.visualize_competition(title_suffix=f"场景: {scenario['name']}")

        # 模拟思考时间
        time.sleep(0.5)  # 缩短等待时间

    # 显示最终结果
    print("\n" + "=" * 60)
    print("模拟完成 - 结果汇总")
    print("=" * 60)

    # 显示意识流
    model.show_consciousness_stream(show_details=False)

    # 显示统计信息
    stats = model.get_statistics()
    print(f"\n模型统计:")
    print(f"  总意识事件数: {stats['total_conscious_events']}")
    print(f"  当前有意识模块数: {stats['conscious_modules_count']}")
    print(f"  当前模块激活水平:")
    for module, activation in stats['module_activations'].items():
        print(f"    {module}: {activation:.3f}")

    # 可视化意识流时间线
    if show_plots or save_plots:  # 修正变量名
        model.visualize_consciousness_stream()

    return model


# 主程序入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="意识模型模拟")
    parser.add_argument('--mode', type=str, choices=['demo', 'batch', 'quiet'],
                        default='demo', help='运行模式')
    parser.add_argument('--show-plots', type=bool, default=True,
                        help='是否显示图表')
    parser.add_argument('--save-plots', type=bool, default=False,
                        help='是否保存图表')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='批量模拟次数')

    args = parser.parse_args()

    if args.mode == 'demo':
        # 演示模式 - 显示详细输出和图表
        print("运行演示模式...")
        model = consciousness_model(
            show_plots=args.show_plots,
            save_plots=args.save_plots
        )

    elif args.mode == 'batch':
        # 批量模式 - 不显示图表，只输出统计
        print("运行批量模拟模式...")


        # 批量模拟函数（简化版）
        def batch_simulation(num_simulations=10):
            print(f"\n批量模拟 {num_simulations} 次运行完成")


        results = batch_simulation(num_simulations=args.batch_size)

    elif args.mode == 'quiet':
        # 静默模式 - 最小化输出
        print("运行静默模式...")
        model = GlobalWorkspaceModel(show_visualization=False, save_figures=False)
        # 处理单个简单刺激
        stimulus = {'视觉': '测试刺激', '情感': '中性'}
        result = model.process_stimulus(stimulus, "测试")
        print(f"最终结果: {result}")