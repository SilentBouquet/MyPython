# 用函数编写一个计算器，能计算加减乘除四则运算
def calculator(first, opt, second):
    if opt == '+':
        return first + second
    elif opt == '-':
        return first - second
    elif opt == '*':
        return first * second
    elif opt == '/':
        return first / second
    else:
        return 'ERROR'


a = eval(input('请输入第一个整数：'))
o = input('请输入运算法则：')
b = eval(input('请输入第二个整数：'))
result = calculator(a, o, b)
print('结果为{}'.format(result))