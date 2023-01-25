import hashlib
import string
import random
from tqdm import tqdm


# gets a sha-1 hash truncated to the first n bits
def my_hash(my_string, n):
    h = int.from_bytes(hashlib.sha1(str.encode(my_string)).digest(), byteorder="big")
    h = h >> (160 - n)
    return hex(h)


def random_string(n=32):
    return "".join(random.choices(string.ascii_letters, k=n))


def pre_image_attack(my_string, n):
    goal = my_hash(my_string, n)
    counter = 0
    match = False
    while not match:
        test = my_hash(random_string(), n)
        counter += 1
        if goal == test:
            match = True
    return counter


def collision_attack(n):
    hash_list = [my_hash(random_string(), n)]
    counter = 0
    match = False
    while not match:
        test = my_hash(random_string(), n)
        counter += 1
        if test in hash_list:
            match = True
        hash_list.append(test)
    return counter


def run_trials(attack_type, num_trials, n, pi_string=None):
    if attack_type == "pi":
        results = []
        assert pi_string is not None
        for _ in tqdm(range(num_trials)):
            results.append(pre_image_attack(pi_string, n))
        return results
    elif attack_type == "c":
        results = []
        for _ in tqdm(range(num_trials)):
            results.append(collision_attack(n))
        return results
    else:
        print("attack_type should be pi or c!")
        return None


if __name__ == '__main__':
    num_bits = 8

    pi_results = run_trials("pi", 50, num_bits, "abc")
    c_results = run_trials("c", 50, num_bits)

    print("pre_image:", sum(pi_results) / len(pi_results))
    print("collision:", sum(c_results) / len(c_results))
