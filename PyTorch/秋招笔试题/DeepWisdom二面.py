import heapq
import threading
import time
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from datetime import datetime
import random


# 任务状态枚举
class TaskStatus(Enum):
    PENDING = "待执行"
    RUNNING = "执行中"
    COMPLETED = "已完成"
    FAILED = "失败"


# 任务类
class Task:
    def __init__(self, id: int, priority: int, payload: Any):
        # 验证优先级范围
        if not 1 <= priority <= 5:
            raise ValueError("优先级必须在1-5之间")

        self.id = id
        self.priority = priority
        self.payload = payload
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now()

    # 为了能够使用优先队列，我们需要实现比较方法
    def __lt__(self, other):
        # 首先比较优先级（数字越小优先级越高）
        if self.priority != other.priority:
            return self.priority < other.priority
        # 如果优先级相同，比较创建时间（先创建的任务优先）
        return self.created_at < other.created_at

    def __repr__(self):
        return f"Task(id={self.id}, priority={self.priority}, status={self.status.value})"


# 任务调度器
class TaskScheduler:
    def __init__(self, max_workers: int = 3):
        # 优先队列
        self._queue = []
        # 任务字典，用于快速查找任务
        self._tasks: Dict[int, Task] = {}
        # 线程池执行器
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        # 锁用于线程安全
        self._lock = threading.Lock()
        # 条件变量用于通知新任务到达
        self._condition = threading.Condition(self._lock)
        # 调度器运行状态
        self._running = False
        # 调度线程
        self._scheduler_thread = None

    def add_task(self, task: Task) -> None:
        """添加任务到调度器"""
        with self._lock:
            if task.id in self._tasks:
                raise ValueError(f"任务ID {task.id} 已存在")

            self._tasks[task.id] = task
            # 将任务添加到优先队列
            heapq.heappush(self._queue, task)
            # 通知调度线程有新任务
            self._condition.notify()

    def get_task_status(self, task_id: int) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.status if task else None

    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        with self._lock:
            return list(self._tasks.values())

    def _process_task(self, task: Task) -> None:
        """处理单个任务"""
        try:
            # 更新任务状态为执行中
            with self._lock:
                task.status = TaskStatus.RUNNING

            # 模拟任务执行 - 这里可以替换为实际的任务处理逻辑
            print(f"开始执行任务 {task.id}: {task.payload}")
            time.sleep(2)  # 模拟任务执行时间

            # 随机模拟成功或失败
            if random.random() < 0.8:  # 80%的成功率
                with self._lock:
                    task.result = f"任务 {task.id} 完成"
                    task.status = TaskStatus.COMPLETED
                print(f"任务 {task.id} 完成")
            else:
                raise RuntimeError("模拟任务执行失败")

        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
            print(f"任务 {task.id} 失败: {e}")

    def _scheduler_loop(self):
        """调度器主循环"""
        while self._running:
            with self._lock:
                # 如果队列为空，等待新任务
                if not self._queue:
                    self._condition.wait()
                    continue

                # 获取优先级最高的任务
                task = heapq.heappop(self._queue)

                # 如果任务已经不是待执行状态，跳过
                if task.status != TaskStatus.PENDING:
                    continue

                # 提交任务到线程池
                self._executor.submit(self._process_task, task)

            # 短暂休眠避免过度占用CPU
            time.sleep(0.1)

    def start(self):
        """启动调度器"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self._scheduler_thread.daemon = True
            self._scheduler_thread.start()
            print("任务调度器已启动")

    def stop(self):
        """停止调度器"""
        with self._lock:
            self._running = False
            self._condition.notify_all()
        self._executor.shutdown(wait=True)
        print("任务调度器已停止")


import time
import random
from datetime import datetime


def complex_test():
    """复杂测试用例，验证调度器在各种边界情况和异常场景下的表现"""
    print("=" * 60)
    print("开始复杂测试")
    print("=" * 60)

    # 创建调度器
    scheduler = TaskScheduler(max_workers=3)

    # 测试1: 优先级边界值测试
    print("\n1. 优先级边界值测试")
    try:
        # 测试优先级为0的任务（应该抛出异常）
        task0 = Task(id=0, priority=0, payload="invalid_priority_0")
        scheduler.add_task(task0)
        print("错误: 优先级0的任务应该抛出异常")
    except ValueError as e:
        print(f"✓ 正确捕获异常: {e}")

    try:
        # 测试优先级为6的任务（应该抛出异常）
        task6 = Task(id=6, priority=6, payload="invalid_priority_6")
        scheduler.add_task(task6)
        print("错误: 优先级6的任务应该抛出异常")
    except ValueError as e:
        print(f"✓ 正确捕获异常: {e}")

    # 测试优先级为1和5的边界任务
    task_min = Task(id=100, priority=1, payload="min_priority")
    task_max = Task(id=101, priority=5, payload="max_priority")
    scheduler.add_task(task_min)
    scheduler.add_task(task_max)
    print("✓ 优先级边界任务添加成功")

    # 测试2: 重复ID测试
    print("\n2. 重复ID测试")
    try:
        duplicate_task = Task(id=1, priority=2, payload="duplicate_id")
        scheduler.add_task(duplicate_task)
        print("错误: 重复ID的任务应该抛出异常")
    except ValueError as e:
        print(f"✓ 正确捕获异常: {e}")

    # 启动调度器
    scheduler.start()

    # 等待一段时间让边界任务执行
    time.sleep(1)

    # 测试3: 并发压力测试
    print("\n3. 并发压力测试")
    # 添加多个任务，测试并发处理能力
    for i in range(10, 25):
        priority = random.randint(1, 5)
        task = Task(id=i, priority=priority, payload=f"concurrent_task_{i}")
        scheduler.add_task(task)

    print("✓ 添加了15个并发任务")

    # 测试4: 任务失败处理测试
    print("\n4. 任务失败处理测试")
    # 添加一些容易失败的任务（设置较短的执行时间但高失败率）
    for i in range(30, 35):
        task = Task(id=i, priority=2, payload=f"likely_to_fail_{i}")
        scheduler.add_task(task)

    print("✓ 添加了5个容易失败的任务")

    # 测试5: 大量任务测试
    print("\n5. 大量任务测试")
    # 添加大量任务，测试调度器的处理能力
    for i in range(100, 150):
        priority = random.randint(1, 5)
        task = Task(id=i, priority=priority, payload=f"mass_task_{i}")
        scheduler.add_task(task)

    print("✓ 添加了50个大量任务")

    # 测试6: 状态查询时机测试
    print("\n6. 状态查询时机测试")
    # 在不同时间点查询任务状态
    for _ in range(5):
        time.sleep(0.5)
        pending_count = 0
        running_count = 0
        completed_count = 0
        failed_count = 0

        for task in scheduler.get_all_tasks():
            if task.status == TaskStatus.PENDING:
                pending_count += 1
            elif task.status == TaskStatus.RUNNING:
                running_count += 1
            elif task.status == TaskStatus.COMPLETED:
                completed_count += 1
            elif task.status == TaskStatus.FAILED:
                failed_count += 1

        print(
            f"状态统计: 待执行={pending_count}, 执行中={running_count}, 已完成={completed_count}, 失败={failed_count}")

    # 等待一段时间让更多任务完成
    time.sleep(5)

    # 测试7: 调度器启停测试
    print("\n7. 调度器启停测试")
    print("停止调度器...")
    scheduler.stop()

    # 尝试在停止后添加任务
    try:
        stopped_task = Task(id=999, priority=3, payload="after_stop")
        scheduler.add_task(stopped_task)
        print("错误: 停止后添加任务应该失败")
    except Exception as e:
        print(f"✓ 正确捕获异常: {e}")

    # 重新启动调度器
    print("重新启动调度器...")
    scheduler.start()

    # 添加新任务
    restart_task = Task(id=1000, priority=1, payload="after_restart")
    scheduler.add_task(restart_task)
    print("✓ 重启后成功添加任务")

    # 等待一段时间让新任务执行
    time.sleep(2)

    # 最终状态统计
    print("\n8. 最终状态统计")
    pending_count = 0
    running_count = 0
    completed_count = 0
    failed_count = 0

    for task in scheduler.get_all_tasks():
        if task.status == TaskStatus.PENDING:
            pending_count += 1
        elif task.status == TaskStatus.RUNNING:
            running_count += 1
        elif task.status == TaskStatus.COMPLETED:
            completed_count += 1
        elif task.status == TaskStatus.FAILED:
            failed_count += 1

    print(f"最终状态统计:")
    print(f"  待执行: {pending_count}")
    print(f"  执行中: {running_count}")
    print(f"  已完成: {completed_count}")
    print(f"  失败: {failed_count}")

    # 打印一些失败任务的错误信息
    failed_tasks = [task for task in scheduler.get_all_tasks() if task.status == TaskStatus.FAILED]
    if failed_tasks:
        print("\n失败任务示例:")
        for task in failed_tasks[:3]:  # 只显示前3个失败任务
            print(f"  任务 {task.id}: {task.error}")

    # 停止调度器
    scheduler.stop()

    print("\n" + "=" * 60)
    print("复杂测试完成")
    print("=" * 60)


# 修改任务处理函数，使其更容易失败（用于测试）
def _process_task(self, task: Task) -> None:
    """处理单个任务（修改版，更容易失败）"""
    try:
        # 更新任务状态为执行中
        with self._lock:
            task.status = TaskStatus.RUNNING

        # 模拟任务执行 - 这里可以替换为实际的任务处理逻辑
        print(f"开始执行任务 {task.id}: {task.payload}")

        # 根据任务内容决定执行时间和失败概率
        if "likely_to_fail" in task.payload:
            # 容易失败的任务：执行时间短但失败率高
            time.sleep(0.5)
            if random.random() < 0.7:  # 70%的失败率
                raise RuntimeError("模拟任务执行失败（高失败率）")
        elif "mass_task" in task.payload:
            # 大量任务：执行时间中等，中等失败率
            time.sleep(1)
            if random.random() < 0.3:  # 30%的失败率
                raise RuntimeError("模拟任务执行失败（中等失败率）")
        else:
            # 普通任务：正常执行时间和失败率
            time.sleep(2)
            if random.random() < 0.2:  # 20%的失败率
                raise RuntimeError("模拟任务执行失败")

        # 如果执行成功
        with self._lock:
            task.result = f"任务 {task.id} 完成于 {datetime.now()}"
            task.status = TaskStatus.COMPLETED
        print(f"任务 {task.id} 完成")

    except Exception as e:
        with self._lock:
            task.status = TaskStatus.FAILED
            task.error = str(e)
        print(f"任务 {task.id} 失败: {e}")


# 替换原有的任务处理函数
TaskScheduler._process_task = _process_task

# 运行复杂测试
if __name__ == "__main__":
    complex_test()