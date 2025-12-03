class Solution:
    def getMin(self, n1, n5, n10, n20, n50, M):
        money_cnt = [n1, n5, n10, n20, n50]
        money = [1, 5, 10, 20, 50]

        def cost(m, index):
            is_payed = False
            while index >= 0:
                if money_cnt[index] == 0:
                    index -= 1
                else:
                    money_cnt[index] -= 1
                    m -= money[index]
                    is_payed = True
                    print(f"消费了{money[index]}，还剩{m}")
                    break
            return is_payed, m

        cnt = 0
        is_payed = False
        while M > 0:
            if M >= 50:
                is_payed, M = cost(M, 4)
            elif M >= 20:
                is_payed, M = cost(M, 3)
            elif M >= 10:
                is_payed, M = cost(M, 2)
            elif M >= 5:
                is_payed, M = cost(M, 1)
            elif M >= 1:
                is_payed, M = cost(M, 0)

            if not is_payed:
                break
            else:
                cnt += 1

        return cnt


if __name__ == '__main__':
    s = Solution()
    print(s.getMin(1, 1, 1, 2, 1, 100))