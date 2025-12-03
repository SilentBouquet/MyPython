class Node:
    def __init__(self, value, father=None):
        self.value = value
        self.child = []
        self.father = father


class Solution:
    def findMatchingPaths(self, nodes, query_chain):
        node_dct = {}
        name_dct = {}
        for node in nodes:
            num = node[0]
            value = node[1]
            father_num = node[2]
            name_dct[value] = num

            if father_num:
                father_node = node_dct[father_num]
                n = Node(value, father_num)
                father_node.child.append(num)
            else:
                n = Node(value)
            node_dct[num] = n

        start_pos = query_chain[0]
        end_pos = query_chain[-1]

        start_num = name_dct[start_pos]
        end_num = name_dct[end_pos]

        start_node = node_dct[start_num]
        end_node = node_dct[end_num]

        pre_path = []
        father_num = start_node.father
        while father_num:
            pre_path.append(father_num)
            father_node = node_dct[father_num]
            father_num = father_node.father

        pre_path.reverse()
        print(pre_path)

        current_path = []
        for i in query_chain:
            num = name_dct[i]
            current_path.append(num)

        print(current_path)

        end_nodes = []
        current_num = end_num
        current_node = node_dct[current_num]
        for i in current_node.child:
            item = [i]
            end_nodes.append(item)

        while True:
            is_child = True

            new_path = []
            for i in end_nodes:
                node = node_dct[i[-1]]
                if len(node.child) > 0:
                    is_child = False
                    for j in node.child:
                        current_path = i.copy()
                        current_path.append(j)
                        new_path.append(current_path)
                else:
                    new_path.append(i)

            if is_child:
                break

            end_nodes = new_path

        print(end_nodes)

        all_paths = []
        now_path = pre_path + current_path
        for i in end_nodes:
            all_path = now_path.copy()
            all_path.extend(i)
            all_paths.append(all_path)

        all_paths.sort(key=len)

        return all_paths