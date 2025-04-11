import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from collections import defaultdict
from collections import namedtuple
import matplotlib.patches as patches

# 常量定义
CELL_SIZE = 100  # 每个网格单元的大小（像素）
MARGIN = 10      # 网格单元内部的边距（像素）# 常量定义


def get_coords(row, col, loc='center'):
    """
    计算网格单元的坐标。
    参数：
        row: 网格单元的行索引
        col: 网格单元的列索引
        loc: 坐标类型（'center'、'interior_corners'、'interior_triangle'）
    返回值：
        根据 loc 返回不同的坐标：
            - 'center': 单元中心坐标
            - 'interior_corners': 单元内部的四个角点
            - 'interior_triangle': 单元内部的三角形顶点
    """
    xc = (col + 1.5) * CELL_SIZE
    yc = (row + 1.5) * CELL_SIZE
    if loc == 'center':
        return xc, yc
    elif loc == 'interior_corners':
        half_size = CELL_SIZE // 2 - MARGIN
        xl, xr = xc - half_size, xc + half_size
        yt, yb = yc - half_size, yc + half_size
        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
    elif loc == 'interior_triangle':
        x1, y1 = xc, yc + CELL_SIZE // 3
        x2, y2 = xc + CELL_SIZE // 3, yc - CELL_SIZE // 3
        x3, y3 = xc - CELL_SIZE // 3, yc - CELL_SIZE // 3
        return [(x1, y1), (x2, y2), (x3, y3)]


def draw_object(ax, coords_list, color):
    """
    在网格单元内绘制图形。
    参数：
        ax: Matplotlib 的绘图轴
        coords_list: 图形的坐标列表
        color: 图形的颜色
    逻辑：
        根据 coords_list 的长度绘制不同的图形：
            - 长度为 1：绘制圆形
            - 长度为 3：绘制三角形
            - 长度大于 3：绘制多边形
    """
    if len(coords_list) == 1:  # 圆形
        circle = patches.Circle(coords_list[0], radius=int(0.45 * CELL_SIZE),
                                edgecolor=color, facecolor=color)
        ax.add_patch(circle)
    elif len(coords_list) == 3:  # 三角形
        triangle = patches.Polygon(coords_list, closed=True,
                                   edgecolor=color, facecolor=color)
        ax.add_patch(triangle)
    elif len(coords_list) > 3:  # 多边形
        polygon = patches.Polygon(coords_list, closed=True,
                                  edgecolor=color, facecolor=color)
        ax.add_patch(polygon)


class GridWorldEnv:
    """
    网格世界环境类，用于定义状态、动作、奖励机制和图形渲染。
    """
    def __init__(self, num_rows=4, num_cols=6, delay=0.1):
        """
        初始化网格世界环境。
        参数：
            num_rows: 网格的行数
            num_cols: 网格的列数
            delay: 渲染延迟时间（秒）
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.delay = delay

        # 定义动作空间（上、下、左、右）
        move_up = lambda row, col: (max(row - 1, 0), col)
        move_down = lambda row, col: (min(row + 1, num_rows - 1), col)
        move_left = lambda row, col: (row, max(col - 1, 0))
        move_right = lambda row, col: (row, min(col + 1, num_cols - 1))

        self.action_defs = {0: move_up, 1: move_right,
                            2: move_down, 3: move_left}

        # 状态空间
        self.nS = num_cols * num_rows
        self.nA = len(self.action_defs)
        # 创建从网格坐标到状态编号的映射字典
        self.grid2state_dict = {(s // num_cols, s % num_cols): s
                                for s in range(self.nS)}
        # 创建从状态编号到网格坐标的映射字典
        self.state2grid_dict = {s: (s // num_cols, s % num_cols)
                                for s in range(self.nS)}

        # 陷阱和金子的位置
        gold_cell = (num_rows // 2, num_cols - 2)
        trap_cells = [((gold_cell[0] + 1), gold_cell[1]),
                      (gold_cell[0], gold_cell[1] - 1),
                      ((gold_cell[0] - 1), gold_cell[1])]

        gold_state = self.grid2state_dict[gold_cell]
        trap_states = [self.grid2state_dict[(r, c)]
                       for (r, c) in trap_cells]
        # 将金子的状态编号和陷阱的状态编号列表合并，形成终端状态列表
        self.terminal_states = [gold_state] + trap_states

        # 构建状态转换矩阵
        self.P = defaultdict(dict)
        for s in range(self.nS):
            row, col = self.state2grid_dict[s]
            self.P[s] = defaultdict(list)
            for a in range(self.nA):
                action = self.action_defs[a]
                next_row, next_col = action(row, col)
                next_s = self.grid2state_dict[(next_row, next_col)]

                # 终端状态处理
                if self.is_terminal(next_s):
                    r = 1.0 if next_s == self.terminal_states[0] else -1.0
                else:
                    r = 0.0
                if self.is_terminal(s):
                    done = True
                    next_s = s
                else:
                    done = False
                self.P[s][a] = [(1.0, next_s, r, done)]

        # 初始状态分布
        self.isd = np.zeros(self.nS)
        self.isd[0] = 1.0

        # 定义动作和观察空间
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.state = None
        self.fig, self.ax = None, None
        self.agent_patch = None

    def is_terminal(self, state):
        """
        判断一个状态是否为终端状态（金子或陷阱）。
        参数：
            state: 状态
        返回值：
            布尔值，True 表示终端状态
        """
        return state in self.terminal_states

    def _build_display(self, ax, gold_cell, trap_cells):
        """
        构建图形界面。
        参数：
            ax: Matplotlib 的绘图轴
            gold_cell: 金子的位置
            trap_cells: 陷阱的位置
        """
        screen_width = (self.num_cols + 2) * CELL_SIZE
        screen_height = (self.num_rows + 2) * CELL_SIZE

        # 绘制网格
        for col in range(self.num_cols + 1):
            x = (col + 1) * CELL_SIZE
            ax.plot([x, x], [CELL_SIZE, (self.num_rows + 1) * CELL_SIZE], 'k-')

        for row in range(self.num_rows + 1):
            y = (row + 1) * CELL_SIZE
            ax.plot([CELL_SIZE, (self.num_cols + 1) * CELL_SIZE], [y, y], 'k-')

        # 绘制边框
        bp_list = [
            (CELL_SIZE - MARGIN, CELL_SIZE - MARGIN),
            (screen_width - CELL_SIZE + MARGIN, CELL_SIZE - MARGIN),
            (screen_width - CELL_SIZE + MARGIN,
             screen_height - CELL_SIZE + MARGIN),
            (CELL_SIZE - MARGIN, screen_height - CELL_SIZE + MARGIN)
        ]
        border = patches.Polygon(bp_list, closed=True, fill=None, edgecolor='k', linewidth=5)
        ax.add_patch(border)

        # 绘制陷阱
        for cell in trap_cells:
            trap_coords = get_coords(*cell, loc='center')
            draw_object(ax, [trap_coords], 'black')

        # 绘制金子
        gold_coords = get_coords(*gold_cell, loc='interior_triangle')
        draw_object(ax, gold_coords, 'yellow')

        # 绘制智能体
        agent_coords = get_coords(0, 0, loc='interior_corners')
        self.agent_patch = patches.Polygon(agent_coords, closed=True,
                                           edgecolor='blue', facecolor='blue')
        ax.add_patch(self.agent_patch)

        ax.set_xlim(0, screen_width)
        ax.set_ylim(0, screen_height)
        ax.set_aspect('equal')

    def reset(self, seed=None, options=None):
        """
        重置环境，将智能体放置在初始状态。
        参数：
            seed: 随机种子
            options: 额外选项
        返回值：
            初始状态和额外信息
        """
        self.state = np.random.choice(len(self.isd), p=self.isd)
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self._build_display(self.ax,
                                (self.num_rows // 2, self.num_cols - 2),
                                [((self.num_rows // 2 + 1), self.num_cols - 2),
                                 (self.num_rows // 2, self.num_cols - 3),
                                 ((self.num_rows // 2 - 1), self.num_cols - 2)])
            plt.xticks([])
            plt.yticks([])
            plt.ion()
            plt.show()
        return self.state, {}

    def step(self, action):
        """
        执行动作，更新智能体的位置。
        参数：
            action: 动作（0: 上, 1: 右, 2: 下, 3: 左）
        返回值：
            新状态、奖励、终止标志、截断标志和额外信息
        """
        row, col = self.state2grid_dict[self.state]
        action_def = self.action_defs[action]
        next_row, next_col = action_def(row, col)
        next_state = self.grid2state_dict[(next_row, next_col)]

        if self.is_terminal(next_state):
            reward = 1.0 if next_state == self.terminal_states[0] else -1.0
            terminated = True
        else:
            reward = 0.0
            terminated = False

        truncated = False
        self.state = next_state
        return next_state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        """
        更新图形界面，显示智能体的位置。
        参数：
            mode: 渲染模式（默认为 'human'）
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self._build_display(self.ax,
                                (self.num_rows // 2, self.num_cols - 2),
                                [((self.num_rows // 2 + 1), self.num_cols - 2),
                                 (self.num_rows // 2, self.num_cols - 3),
                                 ((self.num_rows // 2 - 1), self.num_cols - 2)])
            plt.xticks([])
            plt.yticks([])
            plt.ion()
            plt.show()

        row, col = self.state2grid_dict[self.state]
        agent_coords = get_coords(row, col, loc='interior_corners')

        # 更新智能体位置
        self.agent_patch.set_xy(agent_coords)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(self.delay)

    def close(self):
        """
        关闭图形界面。
        """
        if self.fig:
            plt.close(self.fig)
            self.fig = None


import numpy as np
from collections import defaultdict

class Agent(object):
    """
    基于 Q-Learning 的强化学习代理类。
    该代理通过与环境交互学习最优策略。
    """

    def __init__(
            self, env,
            learning_rate=0.01,
            discount_factor=0.9,
            epsilon_greedy=0.9,
            epsilon_min=0.1,
            epsilon_decay=0.95):
        """
        初始化代理的参数和 Q 表。

        参数：
        - env: 环境对象，代理将在其中进行学习。
        - learning_rate: 学习率，控制 Q 值更新的步长。
        - discount_factor: 折扣因子，用于计算未来奖励的权重。
        - epsilon_greedy: 初始探索率，表示选择随机动作的概率。
        - epsilon_min: 探索率的最小值，确保代理不会完全停止探索。
        - epsilon_decay: 探索率的衰减率，用于逐步减少探索。
        """
        self.env = env
        self.lr = learning_rate  # 学习率
        self.gamma = discount_factor  # 折扣因子
        self.epsilon = epsilon_greedy  # 初始探索率
        self.epsilon_min = epsilon_min  # 探索率的最小值
        self.epsilon_decay = epsilon_decay  # 探索率的衰减率

        # 定义 Q 表，每个状态对应一个动作值数组
        # 使用 defaultdict，当访问一个不存在的状态时，自动创建一个默认值（全零数组）
        self.q_table = defaultdict(lambda: np.zeros(self.env.nA))

    def choose_action(self, state):
        """
        根据当前状态选择一个动作，使用 epsilon-greedy 策略。

        参数：
        - state: 当前状态。

        返回：
        - action: 选择的动作。
        """
        if np.random.uniform() < self.epsilon:
            # 探索：随机选择一个动作
            action = np.random.choice(self.env.nA)
        else:
            # 利用：选择当前状态下的最优动作
            q_vals = self.q_table[state]  # 获取当前状态的 Q 值数组

            # permutation 用于生成数组的随机排列，避免在 Q 值相同时总是选择第一个动作
            perm_actions = np.random.permutation(self.env.nA)
            q_vals = [q_vals[a] for a in perm_actions]  # 按随机顺序排列 Q 值
            perm_q_argmax = np.argmax(q_vals)  # 找到 Q 值最大的动作索引
            action = perm_actions[perm_q_argmax]  # 选择 Q 值最大的动作

        return action

    def _learn(self, transition):
        """
        根据一个转移（transition）更新 Q 表。

        参数：
        - transition: 包含当前状态、动作、奖励、下一个状态和是否结束的元组。
        """
        s, a, r, next_s, done = transition  # 解包转移元组
        q_val = self.q_table[s][a]  # 当前状态-动作对的 Q 值

        if done:
            # 如果任务结束，目标 Q 值等于奖励
            q_target = r
        else:
            # 否则，目标 Q 值等于奖励加上折扣因子乘以下一个状态的最大 Q 值
            q_target = r + self.gamma * np.max(self.q_table[next_s])

        # 更新 Q 表：Q(s, a) += learning_rate * (Q_target - Q(s, a))
        self.q_table[s][a] += self.lr * (q_target - q_val)

        # 调整探索率
        self._adjust_epsilon()

    def _adjust_epsilon(self):
        """
        调整探索率，使其随时间衰减，但不低于最小值。
        """
        if self.epsilon > self.epsilon_min:
            # 如果当前探索率大于最小值，则乘以衰减率
            self.epsilon *= self.epsilon_decay


np.random.seed(1)

# namedtuple 是 Python 中的一个数据结构，用于创建具有命名字段的元组。
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def run_qlearning(agent, env, num_episodes=50):
    history = []
    for episode in range(num_episodes):
        # 从 env.reset() 返回的元组中提取状态
        state, _ = env.reset()
        env.render(mode='human')
        final_reward, n_moves = 0.0, 0
        while True:
            action = agent.choose_action(state)
            next_s, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent._learn(Transition(state, action, reward, next_s, done))
            env.render(mode='human')
            state = next_s
            n_moves += 1
            if done:
                final_reward = reward
                break
        history.append((n_moves, final_reward))
        print(f'Episode {episode}: Reward {final_reward:.2} #Moves {n_moves}')

    return history


def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 10))
    ax = fig.add_subplot(2, 1, 1)
    episodes = np.arange(len(history))
    moves = np.array([h[0] for h in history])
    plt.plot(episodes, moves, lw=4,
             marker="o", markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Moves', size=20)

    ax = fig.add_subplot(2, 1, 2)
    rewards = np.array([h[1] for h in history])
    # 绘制最终奖励随回合的变化图，使用阶梯图
    plt.step(episodes, rewards, lw=4)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Final rewards', size=20)
    # plt.savefig('q-learning-history.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    env = GridWorldEnv(num_rows=5, num_cols=5)
    agent = Agent(env)
    history = run_qlearning(agent, env) 
    env.close()

    plot_learning_history(history)