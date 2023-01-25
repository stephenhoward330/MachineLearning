def get_x(i, j):
    x1 = i-1
    x1 = x1 // 3
    x1 *= 3

    x2 = j-1
    x2 = x2 // 3
    x2 += 1

    return x1 + x2


def get_y(i, j):
    y1 = i-1
    y1 = y1 % 3
    y1 *= 3

    y2 = j-1
    y2 = y2 % 3
    y2 += 1

    return y1 + y2


def get_both(i, j):
    return get_x(i, j), get_y(i, j)


if __name__ == '__main__':
    # Ri = j and Gx = y

    print(get_both(3, 3))
    print(get_both(3, 6))
    print(get_both(4, 5))
    print(get_both(8, 2))
    print(get_both(9, 8))
    print(get_both(1, 7))

    # print(get_x(3, 3))
    # print(get_x(2, 4))
    # print(get_x(1, 9))
    # print(get_x(6, 1))
    # print(get_x(6, 6))
    # print(get_x(4, 9))
    # print(get_x(7, 1))
    # print(get_x(8, 4))
    # print(get_x(9, 9))

    # print(get_y(1, 1), "SHOULD BE 1")
    # print()
    #
    # print("SHOULD BE 9")
    # print(get_y(3, 3))
    # print(get_y(3, 6))
    # print(get_y(3, 9))
    # print(get_y(6, 3))
    # print(get_y(6, 6))
    # print(get_y(6, 9))
    # print(get_y(9, 3))
    # print(get_y(9, 6))
    # print(get_y(9, 9))
    # print()
    #
    # print("SHOULD BE 5")
    # print(get_y(2, 2))
    # print(get_y(2, 5))
    # print(get_y(2, 8))
    # print(get_y(5, 2))
    # print(get_y(5, 5))
    # print(get_y(5, 8))
    # print(get_y(8, 2))
    # print(get_y(8, 5))
    # print(get_y(8, 8))
    # print()
    #
    # print(get_y(4, 4))
    # print(get_y(4, 5))
    # print(get_y(4, 6))
    # print(get_y(5, 4))
    # print(get_y(5, 5))
    # print(get_y(5, 6))
    # print(get_y(6, 4))
    # print(get_y(6, 5))
    # print(get_y(6, 6))
