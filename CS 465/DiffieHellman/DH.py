def mod_exp(base, exp, mod):
    a_bin = '{0:b}'.format(exp)  # represent a as a binary string

    temp_num = base
    result = 1
    for bit in a_bin[::-1]:  # iterate over a_bin right to left
        if bit == '1':
            result = (result * temp_num) % mod
        temp_num = (temp_num * temp_num) % mod
    return result


if __name__ == '__main__':
    # 512 bits, found with command 'openssl prime -generate -bits 512 -safe' on linux
    p = 12574638751569033393529116672006613132942233186115684100966788900998314475110477020027689400567457976647074731454779877385946016494728566110831000952120223
    # 512 bits, found with command 'openssl rand -hex 64'
    a = 0xdf5c055a9f150d3a7bf7cea057841ef4877e66d80d1cb400bbd3112adec547615b141c9144081cb0d99080ec78cf8647487e64a09e3fe8af5bd9355b07f086f6

    my_secret = mod_exp(5, a, p)
    their_secret = 2658556319152634784193729234194389092021875908710952768328287007788850073409337405933048663532164259677910244370608499559225230512920134951451390941790024
    print(my_secret)

    shared_key = mod_exp(their_secret, a, p)
    print(shared_key)
