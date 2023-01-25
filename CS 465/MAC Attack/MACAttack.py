# https://codereview.stackexchange.com/questions/37648/python-implementation-of-sha1
def sha1(data, iv=None, pad=True):
    if iv is None:
        iv = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
    bytes = ""

    h0 = iv[0]
    h1 = iv[1]
    h2 = iv[2]
    h3 = iv[3]
    h4 = iv[4]

    for n in range(len(data)):
        if type(data) == list:
            bytes += '{0:08b}'.format(data[n])
        else:
            bytes += '{0:08b}'.format(ord(data[n]))
    if pad:
        bits = bytes + "1"
        p_bits = bits
        # pad until length equals 448 mod 512
        while len(p_bits) % 512 != 448:
            p_bits += "0"
        # append the original length
        p_bits += '{0:064b}'.format(len(bits)-1)
    else:
        p_bits = bytes

    def chunks(l, n):
        return [l[i:i+n] for i in range(0, len(l), n)]

    def rol(n, b):
        return ((n << b) | (n >> (32 - b))) & 0xffffffff

    for c in chunks(p_bits, 512):
        words = chunks(c, 32)
        w = [0]*80
        for n in range(0, 16):
            w[n] = int(words[n], 2)
        for i in range(16, 80):
            w[i] = rol((w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]), 1)

        a = h0
        b = h1
        c = h2
        d = h3
        e = h4

        # Main loop
        for i in range(0, 80):
            if 0 <= i <= 19:
                f = (b & c) | ((~b) & d)
                k = 0x5A827999
            elif 20 <= i <= 39:
                f = b ^ c ^ d
                k = 0x6ED9EBA1
            elif 40 <= i <= 59:
                f = (b & c) | (b & d) | (c & d)
                k = 0x8F1BBCDC
            elif 60 <= i <= 79:
                f = b ^ c ^ d
                k = 0xCA62C1D6

            temp = rol(a, 5) + f + e + k + w[i] & 0xffffffff
            e = d
            d = c
            c = rol(b, 30)
            b = a
            a = temp

        h0 = h0 + a & 0xffffffff
        h1 = h1 + b & 0xffffffff
        h2 = h2 + c & 0xffffffff
        h3 = h3 + d & 0xffffffff
        h4 = h4 + e & 0xffffffff

    return '%08x%08x%08x%08x%08x' % (h0, h1, h2, h3, h4)


if __name__ == '__main__':
    # original is two blocks (1024 bits)
    original_message = [0x4e, 0x6f, 0x20, 0x6f, 0x6e, 0x65, 0x20, 0x68, 0x61, 0x73, 0x20, 0x63, 0x6f, 0x6d, 0x70, 0x6c,
                        0x65, 0x74, 0x65, 0x64, 0x20, 0x6c, 0x61, 0x62, 0x20, 0x32, 0x20, 0x73, 0x6f, 0x20, 0x67, 0x69,
                        0x76, 0x65, 0x20, 0x74, 0x68, 0x65, 0x6d, 0x20, 0x61, 0x6c, 0x6c, 0x20, 0x61, 0x20, 0x30]
    original_padding = [0x00] * 56
    original_padding.insert(0, 0x80)
    original_size = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xf8]  # 504

    # "but give Stephen an A"
    added_message = [0x62, 0x75, 0x74, 0x20, 0x67, 0x69, 0x76, 0x65, 0x20, 0x53, 0x74, 0x65, 0x70, 0x68, 0x65, 0x6e,
                     0x20, 0x61, 0x6e, 0x20, 0x41]
    added_padding = [0x00] * 34
    added_padding.insert(0, 0x80)
    added_size = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xa8]  # 1192
    to_hash = added_message + added_padding + added_size

    print(sha1(to_hash, iv=[0xe384efad, 0xf26767a6, 0x13162142, 0xb5ef0efb, 0xb9d7659a], pad=False))

    full_message = bytes(original_message + original_padding + original_size
                         + added_message)  # + added_padding + added_size)
    print(full_message.hex())
