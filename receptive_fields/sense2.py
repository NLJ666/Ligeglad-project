import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import matplotlib


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
            '身体': {'activation': 0.0, 'content': '', 'threshold': 0.5}  # 新增身体感觉模块
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

    def process_stimulus(self, stimulus, stimulus_name=None):
        """
        处理外部刺激

        参数：
        - stimulus: 字典形式的刺激，如 {'视觉': '红色', '情感': '惊讶'}
        - stimulus_name: 刺激的名称（可选）

        返回：
        - conscious_content: 意识内容，如果没有模块激活则返回None
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
            '情感': 0.3,  # 情感模块更容易进入意识
            '视觉': 0.2,  # 视觉信息优先级较高
            '听觉': 0.2,
            '身体': 0.3,  # 身体感觉也很重要
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
                           '爱', '恨', '喜欢', '讨厌']
        emotion_factor = 0.0
        for word in emotional_words:
            if word in str(content):
                emotion_factor += 0.2

        # 关键词检测
        keywords = {
            '视觉': ['红色', '蓝色', '绿色', '黄色', '白色', '黑色', '光亮', '黑暗',
                     '移动', '静止', '大', '小', '圆形', '方形', '三角形', '人脸',
                     '动物', '车辆', '建筑', '文字'],
            '听觉': ['声音', '音乐', '说话', '噪音', '安静', '响亮', '柔和', '尖锐',
                     '旋律', '节奏', '人声', '自然声', '机械声', '警报声'],
            '身体': ['疼痛', '舒适', '温暖', '寒冷', '触觉', '压力', '运动', '静止',
                     '疲劳', '精力充沛', '饥饿', '饱腹', '口渴', '痒', '麻木'],
            '语言': ['思考', '问题', '答案', '想法', '概念', '理论', '解释', '描述',
                     '分析', '推理', '判断', '决定'],
            '记忆': ['回忆', '过去', '经历', '学习', '知识', '熟悉', '陌生', '重复',
                     '模式', '关联', '上下文'],
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
            print("警告: 无模块达到激活阈值!")
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

    def reset_module_activations(self):
        """重置所有模块的激活水平（模拟注意力转移）"""
        for module_name in self.modules:
            self.modules[module_name]['activation'] = 0.0
            self.modules[module_name]['content'] = ''
        print("已重置所有模块激活水平")


# 交互式测试函数
def interactive_consciousness_model():
    """
    交互式意识模型 - 用户可以输入各种刺激
    """
    print("初始化交互式意识模型...")
    print("=" * 60)

    model = GlobalWorkspaceModel(
        show_visualization=False,  # 交互模式下默认不显示图表
        save_figures=False
    )

    print("可用的认知模块:")
    modules = model.get_available_modules()
    for i, module in enumerate(modules, 1):
        print(f"  {i}. {module}")

    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("1. 输入格式: 模块名:内容 (多个用逗号分隔)")
    print("   例如: 视觉:一个红色的苹果, 情感:惊讶, 听觉:咔嚓声")
    print("2. 特殊命令:")
    print("   '状态' - 显示当前模型状态")
    print("   '历史' - 显示意识流历史")
    print("   '重置' - 重置所有模块激活")
    print("   '退出' - 退出程序")
    print("   '帮助' - 显示此帮助")
    print("=" * 60)

    while True:
        print("\n" + "-" * 40)
        user_input = input("请输入刺激 (或命令): ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
            print("退出程序...")
            break

        elif user_input.lower() in ['状态', 'status', 's']:
            model.show_current_state()
            continue

        elif user_input.lower() in ['历史', 'history', 'h']:
            if model.stream_of_consciousness:
                print("\n意识流历史记录:")
                print("-" * 40)
                for i, entry in enumerate(model.stream_of_consciousness[-5:], 1):  # 显示最后5条
                    print(f"{i}. 时间: {time.ctime(entry['timestamp'])}")
                    print(f"   刺激: {entry['stimulus']}")
                    print(f"   内容: {entry['content']}")
                    print(f"   模块: {entry['modules']}")
                    print()
            else:
                print("暂无意识流记录")
            continue

        elif user_input.lower() in ['重置', 'reset', 'r']:
            model.reset_module_activations()
            continue

        elif user_input.lower() in ['帮助', 'help', '?']:
            print("\n使用说明:")
            print("  输入格式: 模块名:内容 (多个用逗号分隔)")
            print("  例如: 视觉:一个红色的苹果, 情感:惊讶, 听觉:咔嚓声")
            print("\n  可用模块:", ", ".join(modules))
            print("\n  特殊命令: 状态, 历史, 重置, 帮助, 退出")
            continue

        # 解析用户输入的刺激
        try:
            stimulus = {}
            parts = [part.strip() for part in user_input.split(',')]

            for part in parts:
                if ':' in part:
                    module_part, content_part = part.split(':', 1)
                    module = module_part.strip()
                    content = content_part.strip()

                    if module in modules:
                        stimulus[module] = content
                    else:
                        print(f"警告: 未知模块 '{module}'，将忽略")
                        print(f"可用模块: {', '.join(modules)}")
                else:
                    # 如果没有指定模块，默认使用"语言"模块
                    stimulus['语言'] = part.strip()

            if stimulus:
                print(f"\n正在处理输入: {stimulus}")
                result = model.process_stimulus(stimulus, "用户输入")

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


# 示例刺激生成器
def example_stimuli():
    """返回一些示例刺激"""
    examples = [
        {
            'name': '惊喜',
            'stimulus': {'视觉': '突然出现的生日蛋糕', '情感': '惊喜', '听觉': '生日歌'}
        },
        {
            'name': '疼痛',
            'stimulus': {'身体': '手指被针刺痛', '情感': '痛苦', '视觉': '血滴'}
        },
        {
            'name': '美景',
            'stimulus': {'视觉': '夕阳下的金色海滩', '情感': '宁静', '听觉': '海浪声'}
        },
        {
            'name': '思考',
            'stimulus': {'语言': '人工智能是否能有意识', '情感': '好奇', '记忆': '相关文章'}
        },
        {
            'name': '社交',
            'stimulus': {'视觉': '老朋友的笑容', '听觉': '熟悉的笑声', '情感': '温暖'}
        }
    ]
    return examples


# 运行示例演示
def run_example_demo():
    """运行示例演示"""
    print("运行意识模型示例演示...")
    print("=" * 60)

    model = GlobalWorkspaceModel(
        show_visualization=False,
        save_figures=False
    )

    examples = example_stimuli()

    for i, example in enumerate(examples, 1):
        print(f"\n示例 {i}/{len(examples)}: {example['name']}")
        print(f"刺激: {example['stimulus']}")
        print("-" * 40)

        result = model.process_stimulus(example['stimulus'], example['name'])

        if result:
            print(f"\n✅ 产生的意识: {result}")
        else:
            print(f"\n⚠️  意识较弱")

        # 显示当前状态
        print(f"\n当前激活水平:")
        for module_name, module_info in model.modules.items():
            if module_info['activation'] > 0:
                print(f"  {module_name}: {module_info['activation']:.3f}")

        time.sleep(1.5)
        if i < len(examples):
            input("\n按Enter键继续下一个示例...")

    print("\n" + "=" * 60)
    print("演示完成!")
    model.show_current_state()


# 主程序入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="意识模型模拟")
    parser.add_argument('--mode', type=str,
                        choices=['interactive', 'demo', 'example', 'test'],
                        default='interactive', help='运行模式')
    parser.add_argument('--show-plots', type=bool, default=False,
                        help='是否显示图表')
    parser.add_argument('--save-plots', type=bool, default=False,
                        help='是否保存图表')

    args = parser.parse_args()

    print("=" * 60)
    print("全球工作空间意识模型")
    print("=" * 60)

    if args.mode == 'interactive':
        # 交互模式 - 用户可以输入刺激
        print("进入交互模式...")
        interactive_consciousness_model()

    elif args.mode == 'demo':
        # 演示模式 - 运行预定义场景
        print("运行演示模式...")
        from typing import Dict, List, Optional


        def test_enhanced_consciousness_model(show_plots, save_plots):
            model = GlobalWorkspaceModel(
                show_visualization=show_plots,
                save_figures=save_plots
            )

            scenarios = [
                {'name': '紧急情况',
                 'stimulus': {'视觉': '前方突然出现的红色警示灯', '听觉': '刺耳的刹车声', '情感': '危险警报'}},
                {'name': '社交互动',
                 'stimulus': {'视觉': '熟悉的朋友微笑的面孔', '听觉': '温暖的问候声', '情感': '愉悦',
                              '记忆': '之前的美好回忆'}},
                {'name': '学习场景',
                 'stimulus': {'语言': '重要的数学公式推导', '记忆': '相关的知识点', '情感': '专注'}},
                {'name': '自然环境',
                 'stimulus': {'视觉': '阳光下的绿色草地和鲜花', '听觉': '鸟鸣声和风声', '情感': '平静安宁'}}
            ]

            for scenario in scenarios:
                print(f"\n场景: {scenario['name']}")
                result = model.process_stimulus(scenario['stimulus'], scenario['name'])
                if result:
                    print(f"意识: {result}")
                time.sleep(1)

            return model


        model = test_enhanced_consciousness_model(
            show_plots=args.show_plots,
            save_plots=args.save_plots
        )

    elif args.mode == 'example':
        # 示例模式
        run_example_demo()

    elif args.mode == 'test':
        # 测试模式 - 快速测试
        print("运行测试模式...")
        model = GlobalWorkspaceModel(show_visualization=False, save_figures=False)

        # 测试一些输入
        test_inputs = [
            "视觉:一个红色的球, 情感:好奇",
            "身体:头痛, 情感:痛苦",
            "语言:今天的天气真好, 情感:愉快",
            "听觉:远处传来的音乐声, 记忆:熟悉的旋律"
        ]

        for test_input in test_inputs:
            print(f"\n测试输入: {test_input}")
            # 解析输入
            stimulus = {}
            parts = [part.strip() for part in test_input.split(',')]
            for part in parts:
                if ':' in part:
                    module, content = part.split(':', 1)
                    stimulus[module.strip()] = content.strip()

            result = model.process_stimulus(stimulus, "测试")
            print(f"结果: {result if result else '无意识'}")
            time.sleep(0.5)

    print("\n" + "=" * 60)
    print("程序结束")
    print("=" * 60)