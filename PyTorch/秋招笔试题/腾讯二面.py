"""
1、GPU 集群资源调度优化
题目要求：
给定一组 GPU 任务，每个任务有所需的 GPU 数量和预计的执行时间。现有若干个 GPU 集群，每个集群有一定数量的 GPU 卡。
任务可以分配给任意集群，但需满足任务所需的 GPU 数量。目标是找到一种分配方案，使得所有任务完成的总时间最短。
示例1：

输入：
tasks = [(2, 10), (3, 8), (1, 5)]
clusters = [3, 2]

输出：13。
"""


def solve(tasks, clusters):
    L = []

    for i in range(len(tasks)):
        L.append([tasks[i][0], tasks[i][1]])

    while True:
        is_swapped = False

        for i in range(len(L)-1):
            if L[i][1] > L[i+1][1]:
                tm = L[i]
                L[i] = L[i+1]
                L[i+1] = tm
                is_swapped = True

        if not is_swapped:
            break

    clusters.sort()

    start = 0
    used = {}
    while L:
        print("\n当前剩余任务：", L)
        print("当前可用资源：", clusters)
        print("当前占用资源：", used)

        if used:
            for key in used.keys():
                value = used[key]
                value[1] -= 1
                if value[1] == 0:
                    clusters[key] += value[0]

        for task in L:
            cnt = task[0]
            t = task[1]

            for i in range(len(clusters)-1, -1, -1):
                if clusters[i] >= cnt:
                    clusters[i] -= cnt
                    used[i] = [cnt, t]
                    L.remove(task)

        start += 1

    return start


if __name__ == '__main__':
    tasks = [(2, 10), (3, 8), (1, 5)]
    clusters = [3, 2]
    print(solve(tasks, clusters))