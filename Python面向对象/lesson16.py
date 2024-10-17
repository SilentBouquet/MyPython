# 继承补充：mro和c3算法
class D(object):
    pass


class C(D):
    pass


class B(D):
    pass


class A(B, C):
    pass


# mro(A) = [A] + merge(mro(B), merge(C), [B, C])
# mro(A) = [A] + merge([B, C], [C, D], [B, C])
# mro(A) = [A] + [B, C, D]
# mro(A) = [A, B, C, D, object10]
# 补充：从左到右，深度优先，大小钻石，留住顶端
print(A.mro())