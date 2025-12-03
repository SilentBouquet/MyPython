class Solution:
    def bigAdd(self, numStr1, numStr2):
        Result = []

        float_num1 = ""
        float_num2 = ""
        int_num1 = numStr1
        int_num2 = numStr2
        if '.' in numStr1:
            float_num1 = numStr1.split('.')[1]
            int_num1 = numStr1.split('.')[0]
        if '.' in numStr2:
            float_num2 = numStr2.split('.')[1]
            int_num2 = numStr2.split('.')[0]

        p = 0
        if float_num1 and not float_num2:
            Result.append('.' + float_num1)
        elif not float_num1 and float_num2:
            Result.append('.' + float_num2)
        elif float_num1 and float_num2:
            pos1 = pos2 = max(len(float_num2), len(float_num1)) - 1
            while pos1 >= 0 or pos2 >= 0:
                x = int(float_num1[pos1]) if 0 <= pos1 < len(float_num1) else 0
                y = int(float_num2[pos2]) if 0 <= pos2 < len(float_num2) else 0
                num = (x + y + p) % 10
                p = int((x + y + p) / 10)
                if num != 0:
                    Result.append(str(num))
                pos1 -= 1
                pos2 -= 1

            if Result:
                Result.append('.')

        pos1 = len(int_num1) - 1
        pos2 = len(int_num2) - 1
        while pos1 >= 0 or pos2 >= 0:
            x = int(int_num1[pos1]) if pos1 >= 0 else 0
            y = int(int_num2[pos2]) if pos2 >= 0 else 0
            num = (x + y + p) % 10
            p = int((x + y + p) / 10)
            Result.append(str(num))
            pos1 -= 1
            pos2 -= 1

        if p != 0:
            Result.append(str(p))

        return "".join(Result[::-1])


if __name__ == '__main__':
    s = Solution()
    print(s.bigAdd('1.99', '4.01'))