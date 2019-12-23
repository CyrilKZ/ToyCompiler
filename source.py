def fun(s: str, length: int) -> bool:
    i = length - 1
    j = 0
    flag = True
    while j < length:
        if s[i] != s[j]:
            flag = False
            break
        i = i - 1
        j = j + 1
    return flag
