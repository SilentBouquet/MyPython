import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from collections import defaultdict
import matplotlib.patches as patches

# 常量定义
CELL_SIZE = 100  # 每个网格单元的大小（像素）
MARGIN = 10      # 网格单元内部的边距（像素）


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
        nS = num_cols * num_rows
        nA = len(self.action_defs)
        # 创建从网格坐标到状态编号的映射字典
        self.grid2state_dict = {(s // num_cols, s % num_cols): s
                                for s in range(nS)}
        # 创建从状态编号到网格坐标的映射字典
        self.state2grid_dict = {s: (s // num_cols, s % num_cols)
                                for s in range(nS)}

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
        for s in range(nS):
            row, col = self.state2grid_dict[s]
            self.P[s] = defaultdict(list)
            for a in range(nA):
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
        self.isd = np.zeros(nS)
        self.isd[0] = 1.0

        # 定义动作和观察空间
        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)

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


if __name__ == '__main__':
    # 设置 Matplotlib 后端
    matplotlib.use('Qt5Agg')

    # 创建一个 5x6 的网格世界
    env = GridWorldEnv(8, 8)

    # 测试环境
    for i in range(1):
        s, _ = env.reset()  # 重置环境
        env.render()        # 渲染初始状态

        while True:
            # 随机选择动作
            action = np.random.choice(env.action_space.n)
            next_s, reward, terminated, truncated, _ = env.step(action)
            print(f'Action {action}: State {s} -> {next_s}, Reward {reward}, Done {terminated}')

            env.render()  # 更新图形界面
            if terminated or truncated:
                break

    # 关闭环境
    env.close()