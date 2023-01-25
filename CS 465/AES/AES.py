r_con = [0x00000000,
         0x01000000, 0x02000000, 0x04000000, 0x08000000,
         0x10000000, 0x20000000, 0x40000000, 0x80000000,
         0x1B000000, 0x36000000, 0x6C000000, 0xD8000000,
         0xAB000000, 0x4D000000, 0x9A000000, 0x2F000000,
         0x5E000000, 0xBC000000, 0x63000000, 0xC6000000,
         0x97000000, 0x35000000, 0x6A000000, 0xD4000000,
         0xB3000000, 0x7D000000, 0xFA000000, 0xEF000000,
         0xC5000000, 0x91000000, 0x39000000, 0x72000000,
         0xE4000000, 0xD3000000, 0xBD000000, 0x61000000,
         0xC2000000, 0x9F000000, 0x25000000, 0x4A000000,
         0x94000000, 0x33000000, 0x66000000, 0xCC000000,
         0x83000000, 0x1D000000, 0x3A000000, 0x74000000,
         0xE8000000, 0xCB000000, 0x8D000000]


s_box = [[0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76],
         [0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0],
         [0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15],
         [0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75],
         [0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84],
         [0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf],
         [0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8],
         [0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2],
         [0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73],
         [0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb],
         [0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79],
         [0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08],
         [0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a],
         [0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e],
         [0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf],
         [0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]]


inv_s_box = [[0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb],
             [0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb],
             [0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e],
             [0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25],
             [0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92],
             [0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84],
             [0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06],
             [0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b],
             [0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73],
             [0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e],
             [0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b],
             [0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4],
             [0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f],
             [0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef],
             [0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61],
             [0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]]


def ff_add(a, b):
    return (a ^ b) & 0xFF


def xtime(byte):
    do_xor = False
    if byte & 0x80 == 0x80:  # check if the left-most bit is set
        do_xor = True
    byte = byte << 1  # left shift
    if do_xor:
        byte = ff_add(byte, 0x11b)  # xor
    return byte


def ff_multiply(a, b):
    # iteratively build xtime table on a
    table = {0: a}
    for i in range(1, 8):
        table[i] = xtime(table[i-1])

    result = 0x00
    # iterate through b
    for j in range(0, 8):
        if (b >> j) & 0x01 == 0x01:  # check if the jth bit is set
            result = ff_add(result, table[j])
    return result


def sub_word(word):
    b1 = s_box[(word >> 28) & 0xF][(word >> 24) & 0xF]
    b2 = s_box[(word >> 20) & 0xF][(word >> 16) & 0xF]
    b3 = s_box[(word >> 12) & 0xF][(word >> 8) & 0xF]
    b4 = s_box[(word >> 4) & 0xF][word & 0xF]
    return (b1 << 24) + (b2 << 16) + (b3 << 8) + b4


def rot_word(word):
    b1 = (word & 0xFF000000) >> 24
    b2 = word & 0x00FFFFFF
    return (b2 << 8) + b1


def key_expansion(key):
    nk = round(len(key)/4)
    nb = 4
    nr = nk + nb + 2
    w = [0x00000000] * (nb * (nr + 1))

    i = 0

    while i < nk:
        w[i] = (key[4*i] << 24) + (key[4*i+1] << 16) + (key[4*i+2] << 8) + key[4*i+3]
        i += 1

    i = nk

    while i < nb * (nr + 1):
        temp = w[i-1]
        if i % nk == 0:
            temp = sub_word(rot_word(temp)) ^ r_con[round(i/nk)]
        elif nk > 6 and i % nk == 4:
            temp = sub_word(temp)
        w[i] = w[i-nk] ^ temp
        i += 1

    return w


def sub_bytes(s):
    new_state = [[], [], [], []]
    for i in range(len(s)):
        for j in range(len(s[i])):
            new_state[i].append(s_box[(s[i][j] >> 4) & 0xF][s[i][j] & 0xF])

    return new_state


def shift_rows(s):
    new_state = [[], [], [], []]
    for i in range(len(s)):
        for j in range(len(s[i])):
            # shift each row i bytes
            new_state[i].append(s[i][(j + i) % len(s[i])])

    return new_state


def mix_columns(s):
    new_state = [[], [], [], []]
    for c in range(0, 4):
        new_state[0].append(ff_multiply(0x02, s[0][c]) ^ ff_multiply(0x03, s[1][c]) ^ s[2][c] ^ s[3][c])
        new_state[1].append(s[0][c] ^ ff_multiply(0x02, s[1][c]) ^ ff_multiply(0x03, s[2][c]) ^ s[3][c])
        new_state[2].append(s[0][c] ^ s[1][c] ^ ff_multiply(0x02, s[2][c]) ^ ff_multiply(0x03, s[3][c]))
        new_state[3].append(ff_multiply(0x03, s[0][c]) ^ s[1][c] ^ s[2][c] ^ ff_multiply(0x02, s[3][c]))

    return new_state


def add_round_key(s, w):
    new_state = [[], [], [], []]
    for c in range(4):
        new_state[0].append(s[0][c] ^ ((w[c] >> 24) & 0xFF))
        new_state[1].append(s[1][c] ^ ((w[c] >> 16) & 0xFF))
        new_state[2].append(s[2][c] ^ ((w[c] >> 8) & 0xFF))
        new_state[3].append(s[3][c] ^ (w[c] & 0xFF))

    return new_state


def cipher(message, key, key_size):
    # format message and key
    expanded_key = key_expansion(separate_bytes(key, key_size))
    start_state = form_state(separate_bytes(message, 16))

    print("CIPHER (ENCRYPT):")
    print(f"round[ 0].input    {hex(message) : >34}")
    print(f"round[ 0].k_sch    {hex(words_to_num(expanded_key[:4])) : >34}")

    # start encryption
    state = add_round_key(start_state, expanded_key[:4])

    nr = round((len(expanded_key) / 4) - 2)
    for i in range(nr):
        print(f"round[{i+1 : >2}].start    {hex(state_to_num(state)) : >34}")
        state = sub_bytes(state)
        print(f"round[{i+1 : >2}].s_box    {hex(state_to_num(state)) : >34}")
        state = shift_rows(state)
        print(f"round[{i+1 : >2}].s_row    {hex(state_to_num(state)) : >34}")
        state = mix_columns(state)
        print(f"round[{i+1 : >2}].m_col    {hex(state_to_num(state)) : >34}")
        print(f"round[{i+1 : >2}].k_sch    {hex(words_to_num(expanded_key[4*(i+1):4*(i+1)+4])) : >34}")
        state = add_round_key(state, expanded_key[4*(i+1):])

    print(f"round[{nr + 1 : >2}].start    {hex(state_to_num(state)) : >34}")
    state = sub_bytes(state)
    print(f"round[{nr + 1 : >2}].s_box    {hex(state_to_num(state)) : >34}")
    state = shift_rows(state)
    print(f"round[{nr + 1 : >2}].s_row    {hex(state_to_num(state)) : >34}")
    print(f"round[{nr + 1 : >2}].k_sch    {hex(words_to_num(expanded_key[-4:])) : >34}")
    state = add_round_key(state, expanded_key[-4:])

    print(f"round[{nr + 1 : >2}].output   {hex(state_to_num(state)) : >34}\n")
    return state_to_num(state)


def inv_sub_bytes(s):
    new_state = [[], [], [], []]
    for i in range(len(s)):
        for j in range(len(s[i])):
            new_state[i].append(inv_s_box[(s[i][j] >> 4) & 0xF][s[i][j] & 0xF])

    return new_state


def inv_shift_rows(s):
    new_state = [[], [], [], []]
    for i in range(len(s)):
        for j in range(len(s[i])):
            # shift each row i bytes
            new_state[i].append(s[i][(j - i) % len(s[i])])

    return new_state


def inv_mix_columns(s):
    new_state = [[], [], [], []]
    for c in range(0, 4):
        new_state[0].append(ff_multiply(0x0e, s[0][c]) ^ ff_multiply(0x0b, s[1][c])
                            ^ ff_multiply(0x0d, s[2][c]) ^ ff_multiply(0x09, s[3][c]))
        new_state[1].append(ff_multiply(0x09, s[0][c]) ^ ff_multiply(0x0e, s[1][c])
                            ^ ff_multiply(0x0b, s[2][c]) ^ ff_multiply(0x0d, s[3][c]))
        new_state[2].append(ff_multiply(0x0d, s[0][c]) ^ ff_multiply(0x09, s[1][c])
                            ^ ff_multiply(0x0e, s[2][c]) ^ ff_multiply(0x0b, s[3][c]))
        new_state[3].append(ff_multiply(0x0b, s[0][c]) ^ ff_multiply(0x0d, s[1][c])
                            ^ ff_multiply(0x09, s[2][c]) ^ ff_multiply(0x0e, s[3][c]))

    return new_state


def inv_cipher(cipher_text, key, key_size):
    # format message and key
    expanded_key = key_expansion(separate_bytes(key, key_size))
    start_state = form_state(separate_bytes(cipher_text, 16))

    print("INVERSE CIPHER (DECRYPT):")
    print(f"round[ 0].iinput   {hex(cipher_text) : >34}")
    print(f"round[ 0].ik_sch   {hex(words_to_num(expanded_key[-4:])) : >34}")

    # start decryption
    state = add_round_key(start_state, expanded_key[-4:])
    print(f"round[ 1].istart   {hex(state_to_num(state)) : >34}")
    state = inv_shift_rows(state)
    print(f"round[ 1].is_row   {hex(state_to_num(state)) : >34}")
    state = inv_sub_bytes(state)
    print(f"round[ 1].is_box   {hex(state_to_num(state)) : >34}")

    nr = round((len(expanded_key) / 4) - 2)
    for i in range(nr):
        print(f"round[{i+1 : >2}].ik_sch   {hex(words_to_num(expanded_key[(-4 * (i + 1)) - 4:(-4 * (i + 1))])) : >34}")
        state = add_round_key(state, expanded_key[(-4 * (i + 1)) - 4:])
        print(f"round[{i+1 : >2}].ik_add   {hex(state_to_num(state)) : >34}")
        state = inv_mix_columns(state)
        print(f"round[{i+2 : >2}].istart   {hex(state_to_num(state)) : >34}")
        state = inv_shift_rows(state)
        print(f"round[{i + 2 : >2}].is_row   {hex(state_to_num(state)) : >34}")
        state = inv_sub_bytes(state)
        print(f"round[{i + 2 : >2}].is_box   {hex(state_to_num(state)) : >34}")

    print(f"round[{nr + 1 : >2}].ik_sch   {hex(words_to_num(expanded_key[:4])) : >34}")
    state = add_round_key(state, expanded_key[:4])

    print(f"round[{nr + 1 : >2}].ioutput  {hex(state_to_num(state)) : >34}\n")
    return state_to_num(state)


# ######## HELPER FUNCTIONS ##########

# separates a number into a list of its bytes
def separate_bytes(num, num_bytes):
    byte_list = []
    for _ in range(num_bytes):
        byte_list.insert(0, num & 0xFF)
        num = num >> 8

    return byte_list


# forms a 2d state from a list of bytes
def form_state(bl):
    new_state = [[], [], [], []]
    for i in range(len(new_state)):
        new_state[0].append(bl[i * 4])
        new_state[1].append(bl[(i * 4) + 1])
        new_state[2].append(bl[(i * 4) + 2])
        new_state[3].append(bl[(i * 4) + 3])

    return new_state


# converts a 2d state to a number
def state_to_num(s):
    result = 0
    for i in range(len(s)):
        result += s[3][-(i+1)] << 0 + (32 * i)
        result += s[2][-(i+1)] << 8 + (32 * i)
        result += s[1][-(i+1)] << 16 + (32 * i)
        result += s[0][-(i+1)] << 24 + (32 * i)

    return result


# converts a list of words to a number
def words_to_num(w):
    result = 0
    for i in range(len(w)):
        result += w[i] << (32 * (len(w) - 1 - i))

    return result


# prints a 2d state in hex values
def print_hex(s):
    print([[hex(a) for a in r] for r in s])


def main():
    key = 0x000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
    message_to_encrypt = 0x00112233445566778899aabbccddeeff
    key_size = round(256 / 8)

    encrypted = cipher(message_to_encrypt, key, key_size)

    inv_cipher(encrypted, key, key_size)


if __name__ == '__main__':
    main()
