def mod_exp(base, exp, mod):
    a_bin = '{0:b}'.format(exp)  # represent a as a binary string

    temp_num = base
    result = 1
    for bit in a_bin[::-1]:  # iterate over a_bin right to left
        if bit == '1':
            result = (result * temp_num) % mod
        temp_num = (temp_num * temp_num) % mod
    return result


def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


def extended_gcd(N, a):
    (old_r, r) = (N, a)
    (old_s, s) = (1, 0)
    (old_t, t) = (0, 1)

    while r != 0:
        quotient = old_r // r
        (old_r, r) = (r, old_r - quotient * r)
        (old_s, s) = (s, old_s - quotient * s)
        (old_t, t) = (t, old_t - quotient * t)

    # print("Bézout coefficients:", (old_s, old_t))
    # print("greatest common divisor:", old_r)
    # print("quotients by the gcd:", (t, s))

    if old_t < 0:
        old_t += N

    return old_t


if __name__ == '__main__':
    # 512 bits each, found with command 'openssl prime -generate -bits 512' on linux
    p = 12667639986872036999751711824394770429277604149203666310182828729541933587277812513282010748339098217196003573580788317866140482994683954926035157232565431
    q = 11384746870659521946435186072931151040705992145112766834247052449113662248362563212763951847376074222656588997686154006299481063916481744470048057448440293

    e = 65537

    # if p >> 511:
    #     print("bit set")

    n = p * q
    print("n =  ", n)
    phi = (p-1) * (q-1)
    print("phi =", phi)

    # phi and e should be relatively prime
    assert gcd(phi, e) == 1

    d = extended_gcd(phi, e)
    print("d =", d)

    m = 120527703873533101519616850252758477732725771196559087496384822996112035648094035367321161090959411777274065324273254841589880937913109643437107245623742006364523035004584979623365709240212949766487671494394501746731650473617015341742167108940906659061384758169368294064373395879397198190652754037863672415683
    print(mod_exp(m, e, n))

    m2 = 137030778664857655735464660056914187651386383330929407952073504411159176592817448704166203830938412703228111163610306115380800755029310144320772949597168873336341272085070107708743841548284644082437733542516324264456360949333828593162135583130897890770119555916779253280354535559173961242476428019318016187475
    print(mod_exp(m2, d, n))
