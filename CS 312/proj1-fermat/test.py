
def match_index(array, b, e):  # b is the beginning index, e is the end index
    index = (b + e) // 2
    if array[index] == index:  # we did it!
        return True
    if b == e:  # we failed :(
        return False
    elif array[index] > index:  # go left
        return match_index(array, b, index)
    else:  # go right
        return match_index(array, index+1, e)


if __name__ == '__main__':
    a = [-5, -2, -1, 0, 2, 3, 5, 6, 8, 10]
    print(match_index(a, 0, 9))
