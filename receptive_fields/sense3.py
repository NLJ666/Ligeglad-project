import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import matplotlib
import sys
import os


class GlobalWorkspaceModel:
    """
    基于全球工作空间理论（GWT）的意识模型实现
    """

    def __init__(self, num_modules=5, workspace_capacity=3,
                 show_visualization=True, save_figures=False):
        """
        初始化模型
        """
        # 设置中文字体
        self._setup_fonts()

        self.num_modules = num_modules
        self.workspace_capacity = workspace_capacity
        self.show_visualization = show_visualization
        self.save_figures = save_figures
        self.figure_counter = 0

        # 初始化模块
        self.modules = {
            '视觉': {'activation': 0.0, 'content': '', 'threshold': 0.5},
            '听觉': {'activation': 0.0, 'content': '', 'threshold': 0.5},
            '记忆': {'activation': 0.0, 'content': '', 'threshold': 0.6},
            '语言': {'activation': 0.0, 'content': '', 'threshold': 0.5},
            '情感': {'activation': 0.0, 'content': '', 'threshold': 0.4},
            '身体': {'activation': 0.0, 'content': '', 'threshold': 0.5}
        }

        # 全球工作空间
        self.global_workspace = deque(maxlen=workspace_capacity)

        # 意识流记录
        self.stream_of_consciousness = []

        # 注意力权重
        self.attention_weights = np.random.dirichlet(np.ones(num_modules))

    def _setup_fonts(self):
        """设置matplotlib字体，避免中文乱码"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def process_stimulus(self, stimulus, stimulus_name=None, show_chart=True):
        """
        处理外部刺激
        """
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

        # 显示激活图表
        if show_chart and (self.show_visualization or self.save_figures):
            self.visualize_competition(title_suffix=f"刺激: {stimulus_name or '用户输入'}")

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
        base_activation = np.random.uniform(0.4, 0.8)

        # 模块特定增强
        module_boost = {
            '情感': 0.3,
            '视觉': 0.2,
            '听觉': 0.2,
            '身体': 0.3,
            '语言': 0.1,
            '记忆': 0.1
        }

        if module_name in module_boost:
            base_activation += module_boost[module_name]

        # 内容长度影响
        content_length = len(str(content))
        if content_length > 0:
            content_length_factor = min(content_length / 10, 1.5)
        else:
            content_length_factor = 0.5

        # 情感词汇检测
        emotional_words = ['危险', '警报', '惊讶', '快乐', '愉悦', '悲伤',
                           '专注', '平静', '安宁', '恐惧', '愤怒', '高兴',
                           '中性', '痛苦', '兴奋', '无聊', '紧张', '放松',
                           '爱', '恨', '喜欢', '讨厌', '难过', '同情', '害怕']
        emotion_factor = 0.0
        for word in emotional_words:
            if word in str(content):
                emotion_factor += 0.2

        # 关键词检测
        keywords = {
            '视觉': ['红色', '蓝色', '绿色', '黄色', '白色', '黑色', '光亮', '黑暗',
                     '移动', '静止', '大', '小', '圆形', '方形', '三角形', '人脸',
                     '动物', '车辆', '建筑', '文字', '前方', '突然', '熟悉', '阳光',
                     '草地', '鲜花', '微笑', '面孔', '小孩', '警示灯', '朋友'],
            '听觉': ['声音', '音乐', '说话', '噪音', '安静', '响亮', '柔和', '尖锐',
                     '旋律', '节奏', '人声', '自然声', '机械声', '警报声', '刺耳',
                     '刹车声', '温暖', '问候声', '鸟鸣', '风声', '哭声', '笑声'],
            '身体': ['疼痛', '舒适', '温暖', '寒冷', '触觉', '压力', '运动', '静止',
                     '疲劳', '精力充沛', '饥饿', '饱腹', '口渴', '痒', '麻木'],
            '语言': ['思考', '问题', '答案', '想法', '概念', '理论', '解释', '描述',
                     '分析', '推理', '判断', '决定', '数学', '公式', '推导'],
            '记忆': ['回忆', '过去', '经历', '学习', '知识', '熟悉', '陌生', '重复',
                     '模式', '关联', '上下文', '之前', '美好', '知识点'],
            '情感': emotional_words
        }

        keyword_factor = 0.0
        if module_name in keywords:
            for word in keywords[module_name]:
                if word in str(content):
                    keyword_factor += 0.15

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

        print(f"\n竞争获胜的模块 ({len(competing_modules)}个):")
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

        print(f"\n全球工作空间内容: {broadcast_content}")

        conscious_experience = self._integrate_experience(broadcast_content)
        print(f"\n🌊 意识体验: {conscious_experience}")

        return conscious_experience

    def _integrate_experience(self, broadcast_content):
        """整合多模态信息"""
        experience = []

        # 优先级顺序
        priority_order = ['情感', '身体', '视觉', '听觉', '语言', '记忆']

        for module in priority_order:
            if module in broadcast_content:
                content = broadcast_content[module]
                if module == '情感':
                    experience.append(f"[{content}的感觉]")
                elif module == '身体':
                    experience.append(f"身体: {content}")
                elif module == '视觉':
                    experience.append(f"看到: {content}")
                elif module == '听觉':
                    experience.append(f"听到: {content}")
                elif module == '语言':
                    experience.append(f"思考: {content}")
                else:
                    experience.append(f"回忆: {content}")

        return " | ".join(experience)

    def get_available_modules(self):
        """获取可用的模块列表"""
        return list(self.modules.keys())

    def show_current_state(self):
        """显示当前模型状态"""
        print("\n" + "=" * 60)
        print("当前模型状态:")
        print("=" * 60)

        print(f"\n模块激活状态:")
        for module_name, module_info in self.modules.items():
            threshold = module_info['threshold']
            activation = module_info['activation']
            status = "✅" if activation >= threshold else "❌"
            content = module_info['content'][:20] + "..." if len(module_info['content']) > 20 else module_info[
                'content']
            print(f"  {module_name}: {activation:.3f}/{threshold} {status} 内容: '{content}'")

        print(f"\n全球工作空间内容 ({len(self.global_workspace)}/{self.workspace_capacity}):")
        if self.global_workspace:
            for i, item in enumerate(self.global_workspace):
                print(f"  {i + 1}. {item['module']}: '{item['content']}' (激活: {item['activation']:.3f})")
        else:
            print("  (空)")

        print(f"\n意识流记录数: {len(self.stream_of_consciousness)}")
        if self.stream_of_consciousness:
            print("最近3次意识体验:")
            for i, entry in enumerate(self.stream_of_consciousness[-3:], 1):
                print(f"  {i}. {entry['content']}")

    def reset_module_activations(self):
        """重置所有模块的激活水平"""
        for module_name in self.modules:
            self.modules[module_name]['activation'] = 0.0
            self.modules[module_name]['content'] = ''
        print("已重置所有模块激活水平")

    def visualize_competition(self, title_suffix=""):
        """可视化模块竞争过程"""
        modules = list(self.modules.keys())
        activations = [self.modules[m]['activation'] for m in modules]
        thresholds = [self.modules[m]['threshold'] for m in modules]

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(modules))
        width = 0.6

        # 创建条形图
        colors = ['lightgray'] * len(modules)
        for i, (module, activation) in enumerate(zip(modules, activations)):
            if activation >= self.modules[module]['threshold']:
                colors[i] = 'orange'

        bars = ax.bar(x, activations, width,
                      color=colors, edgecolor='black', linewidth=1.5)

        # 添加阈值线
        ax.plot(x, thresholds, 'r--', label='意识阈值', linewidth=2.5, alpha=0.8)

        # 设置图表属性
        ax.set_xlabel('认知模块', fontsize=12, fontweight='bold')
        ax.set_ylabel('激活水平', fontsize=12, fontweight='bold')

        title = '全球工作空间竞争模型'
        if title_suffix:
            title += f"\n{title_suffix}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        ax.set_xticks(x)
        ax.set_xticklabels(modules, rotation=45, fontsize=11, fontweight='bold')

        # 设置y轴范围
        ax.set_ylim(0, 1.6)
        ax.set_yticks(np.arange(0, 1.7, 0.2))

        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 添加图例
        ax.legend(fontsize=10, loc='upper right')

        # 在条形上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 添加说明文本
        conscious_modules = [m for m, a in zip(modules, activations)
                             if a >= self.modules[m]['threshold']]
        if conscious_modules:
            info_text = f"进入意识的模块: {', '.join(conscious_modules)}"
            box_color = 'lightgreen'
        else:
            info_text = "无模块进入意识"
            box_color = 'lightcoral'

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7))

        plt.tight_layout()

        # 保存或显示图形
        if self.save_figures:
            filename = f"consciousness_model_{self.figure_counter:02d}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 图表已保存为: {filename}")
            self.figure_counter += 1

        if self.show_visualization:
            print("📈 显示激活图表...")
            plt.show(block=False)  # 非阻塞显示
            plt.pause(0.1)  # 短暂暂停，让图表显示
        else:
            plt.close(fig)

        return fig


# 交互式测试函数
def interactive_consciousness_model(show_plots=True, save_plots=False):
    """
    交互式意识模型 - 用户可以输入各种刺激
    """
    print("\n" + "=" * 60)
    print("交互式意识模型")
    print("=" * 60)

    model = GlobalWorkspaceModel(
        show_visualization=show_plots,
        save_figures=save_plots
    )

    modules = model.get_available_modules()

    print("可用的认知模块:")
    for i, module in enumerate(modules, 1):
        threshold = model.modules[module]['threshold']
        print(f"  {i}. {module} (阈值: {threshold})")

    print("\n📋 使用说明:")
    print(" 1. 输入格式: 模块名:内容 (多个用逗号分隔)")
    print("     例如: 视觉:一个红色的苹果, 情感:惊讶, 听觉:咔嚓声")
    print(" 2. 特殊命令:")
    print("     '状态' 或 's' - 显示当前模型状态")
    print("     '历史' 或 'h' - 显示意识流历史")
    print("     '重置' 或 'r' - 重置所有模块激活")
    print("     '图表' 或 'c' - 显示当前激活图表")
    print("     '帮助' 或 '?' - 显示此帮助")
    print("     '退出' 或 'q' - 退出程序")
    print(" 3. 可用模块:", ", ".join(modules))
    print("=" * 60)

    while True:
        print("\n" + "-" * 40)
        user_input = input("请输入刺激 (或命令): ").strip()

        if not user_input:
            continue

        # 处理特殊命令
        cmd = user_input.lower()

        if cmd in ['退出', 'exit', 'quit', 'q']:
            print("退出程序...")
            break

        elif cmd in ['状态', 'status', 's']:
            model.show_current_state()
            continue

        elif cmd in ['历史', 'history', 'h']:
            if model.stream_of_consciousness:
                print("\n意识流历史记录 (最近5次):")
                print("-" * 40)
                for i, entry in enumerate(model.stream_of_consciousness[-5:], 1):
                    print(f"{i}. 时间: {time.ctime(entry['timestamp'])}")
                    print(f"   刺激: {entry['stimulus']}")
                    print(f"   内容: {entry['content']}")
                    print(f"   模块: {entry['modules']}")
                    print()
            else:
                print("暂无意识流记录")
            continue

        elif cmd in ['重置', 'reset', 'r']:
            model.reset_module_activations()
            continue

        elif cmd in ['图表', 'chart', 'c']:
            if show_plots or save_plots:
                model.visualize_competition(title_suffix="用户请求")
            else:
                print("图表显示已禁用，请在启动时使用 --show-plots True")
            continue

        elif cmd in ['帮助', 'help', '?']:
            print("\n📋 使用说明:")
            print("  输入格式: 模块名:内容 (多个用逗号分隔)")
            print("  例如: 视觉:一个红色的苹果, 情感:惊讶, 听觉:咔嚓声")
            print("\n  可用模块:", ", ".join(modules))
            print("\n  特殊命令: 状态(s), 历史(h), 重置(r), 图表(c), 帮助(?), 退出(q)")
            continue

        # 解析用户输入的刺激
        try:
            stimulus = {}
            parts = [part.strip() for part in user_input.split(',')]
            unknown_modules = []

            for part in parts:
                if ':' in part:
                    module_part, content_part = part.split(':', 1)
                    module = module_part.strip()
                    content = content_part.strip()

                    if module in modules:
                        stimulus[module] = content
                    else:
                        unknown_modules.append(module)
                else:
                    # 如果没有指定模块，默认使用"语言"模块
                    stimulus['语言'] = part.strip()

            if unknown_modules:
                print(f"⚠️  警告: 未知模块 {unknown_modules}，将忽略")
                print(f"可用模块: {', '.join(modules)}")

            if stimulus:
                print(f"\n正在处理输入: {stimulus}")
                result = model.process_stimulus(stimulus, "用户输入", show_chart=True)

                if result:
                    print(f"\n✨ 产生的意识: {result}")
                else:
                    print("\n⚠️  没有产生明显的意识体验")
                    print("提示: 尝试使用更强烈或更具体的内容，或包含情感词汇")
            else:
                print("错误: 没有有效的刺激输入")

        except Exception as e:
            print(f"输入解析错误: {e}")
            print("请使用格式: 模块名:内容 (如: 视觉:红色, 情感:惊讶)")

        # 短暂暂停
        time.sleep(0.5)


# 演示模式函数
def test_enhanced_consciousness_model(show_plots=True, save_plots=False):
    """
    演示模式 - 运行预定义场景
    """
    print("\n初始化意识模型...")
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
            'stimulus': {
                '视觉': '阳光下的绿色草地和鲜花',
                '听觉': '鸟鸣声和风声',
                '情感': '平静安宁'
            }
        }
    ]

    print("\n" + "=" * 60)
    print("开始意识模拟实验")
    print("=" * 60)

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'=' * 60}")
        print(f"场景 #{i}: {scenario['name']}")
        print('=' * 60)

        # 处理刺激，并显示图表
        conscious_content = model.process_stimulus(
            scenario['stimulus'],
            scenario['name'],
            show_chart=True
        )

        if conscious_content:
            print(f"\n✅ 产生的意识: {conscious_content}")
        else:
            print(f"\n⚠️  没有产生明显的意识体验")

        # 显示模块激活详情
        print(f"\n模块激活详情:")
        for module_name, module_info in model.modules.items():
            if module_info['activation'] > 0:
                threshold = module_info['threshold']
                activation = module_info['activation']
                status = "✓" if activation >= threshold else "✗"
                print(f"  {module_name}: {activation:.3f}/{threshold} {status}")

        # 等待用户按Enter继续
        if i < len(scenarios):
            input(f"\n按Enter键继续下一个场景...")

    # 显示最终总结
    print("\n" + "=" * 60)
    print("模拟完成 - 结果汇总")
    print("=" * 60)

    print(f"\n总意识事件数: {len(model.stream_of_consciousness)}")
    if model.stream_of_consciousness:
        print("\n所有意识体验:")
        for i, entry in enumerate(model.stream_of_consciousness, 1):
            print(f"  {i}. {entry['content']}")

    return model


# 主程序入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="全球工作空间意识模型模拟",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            使用示例:
              python sense3.py                    # 交互模式 (默认)
              python sense3.py --mode demo       # 演示模式
              python sense3.py --mode test       # 测试模式
              python sense3.py --mode demo --show-plots True   # 显示图表
              python sense3.py --mode demo --show-plots False  # 不显示图表
              python sense3.py --save-plots True               # 保存图表到文件
            """
    )

    parser.add_argument('--mode', type=str,
                        choices=['interactive', 'demo', 'test'],
                        default='interactive', help='运行模式 (默认: interactive)')
    parser.add_argument('--show-plots', type=bool, default=True,
                        help='是否显示图表 (默认: True)')
    parser.add_argument('--save-plots', type=bool, default=False,
                        help='是否保存图表到文件 (默认: False)')

    args = parser.parse_args()

    print("=" * 60)
    print("全球工作空间意识模型")
    print("=" * 60)
    print(f"模式: {args.mode}, 显示图表: {args.show_plots}, 保存图表: {args.save_plots}")

    if args.mode == 'interactive':
        # 交互模式
        interactive_consciousness_model(
            show_plots=args.show_plots,
            save_plots=args.save_plots
        )

    elif args.mode == 'demo':
        # 演示模式
        test_enhanced_consciousness_model(
            show_plots=args.show_plots,
            save_plots=args.save_plots
        )

    elif args.mode == 'test':
        # 测试模式
        print("\n运行测试模式...")
        model = GlobalWorkspaceModel(
            show_visualization=args.show_plots,
            save_figures=args.save_plots
        )

        # 测试一些示例刺激
        test_cases = [
            "视觉:前方突然出现红色的警示灯和浓烟,听觉:刺耳的刹车声和人群尖叫声,情感:极度的恐惧和紧张,身体:心跳加速手心出汗,语言:需要立即采取应急措施,记忆:想起上次车祸的恐怖经历",
            "视觉:夕阳下金色的沙滩和蓝色的大海,听觉:海浪拍打声和海鸥叫声,情感:深深的宁静和幸福,身体:温暖的海风吹过皮肤的舒适感,语言:这就是我一直寻找的宁静之地,记忆:童年时和家人在海边的快乐时光",
            "视觉:复杂的数学公式和图表在屏幕上滚动,听觉:老师清晰讲解的声音和翻书声,情感:专注和好奇心,身体:长时间坐着导致的腰背酸痛,语言:理解了这个概念就能解决整个问题,记忆:上周学过的相关知识点"
            '视觉:熟悉的老朋友温暖的笑容和热情的拥抱,听觉:真诚的问候声和愉快的笑声,情感:温暖和感动,身体:拥抱时的触感和温度,语言:好久不见你最近过得怎么样,记忆:大学时期一起度过的美好时光',
            '视觉:白色的医院墙壁和闪烁的医疗设备,听觉:心电监护仪的嘟嘟声和医生低语,情感:担心和希望,身体:手术后的疼痛和虚弱感,语言:医生说的专业术语需要理解,记忆:去年类似病症的治疗过程'
        ]

        for i, test_input in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}: {test_input}")
            stimulus = {}
            parts = [part.strip() for part in test_input.split(',')]
            for part in parts:
                if ':' in part:
                    module, content = part.split(':', 1)
                    stimulus[module.strip()] = content.strip()

            result = model.process_stimulus(stimulus, f"测试{i}", show_chart=True)
            print(f"结果: {result if result else '无意识'}")
            time.sleep(1)

    print("\n" + "=" * 60)
    print("程序结束")
    print("=" * 60)