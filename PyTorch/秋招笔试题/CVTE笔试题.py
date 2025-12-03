import threading
import time

TASKS = 3                 # 任务数
COUNT = 100               # 百分比
INTERVAL = 1.0            # 每两次打印间隔

# 共享变量
progress = [COUNT] * TASKS        # 各任务剩余百分比
lock = threading.Condition()      # 控制顺序
turn = 0                          # 当前该谁打印


def worker(tid: int):
    global turn
    for _ in range(COUNT, -1, -1):          # 100 … 0
        with lock:
            # 等待轮到自己
            while turn != tid:
                lock.wait()
            # 打印
            print(f"Task{tid+1}: {progress[tid]:3d}%")
            # 更新状态
            progress[tid] -= 1
            turn = (turn + 1) % TASKS
            lock.notify_all()
        time.sleep(INTERVAL)                # 保证两次打印间隔


threads = [threading.Thread(target=worker, args=(i,)) for i in range(TASKS)]
for t in threads:
    t.start()
for t in threads:
    t.join()