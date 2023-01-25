import random


def prime_test(N, k):
    # This is the main function connected to the Test button. You don't need to touch it.
    return run_fermat(N,k), run_miller_rabin(N,k)


# Time complexity of O(n^3), only z is stored so space complexity of O(n)
def mod_exp(x, y, N):  # at most n recursive calls
    if y == 0:  # base case
        return 1
    z = mod_exp(x, y//2, N)  # recursive call, a bit shift is O(n) speed
    if y % 2 == 0:  # if exponent is even
        return (z**2) % N  # multiplication is O(n^2) speed
    else:
        return x*(z**2) % N  # multiplication is O(n^2) speed
    

# Time complexity of O(n^3), space complexity of O(1)
def fprobability(k):
    # You will need to implement this function and change the return value.   
    return 1-(1.0/2.0**k)  # division has complexity O(n^3), multiplication has complexity O(n^2)


# Time complexity of O(n^3), space complexity of O(1)
def mprobability(k):
    # You will need to implement this function and change the return value.   
    return 1-(1.0/4.0)**k  # division has complexity O(n^3), multiplication has complexity O(n^2)


# Time complexity is O(kn^3), space complexity if O(k) because we have k random numbers
def run_fermat(N,k):
    rands = random.sample(range(1, N-1), k)  # get k random numbers from a to N-1, random.sample is O(n) speed
    for a in rands:  # loop through the k random numbers
        if mod_exp(a, N-1, N) != 1:  # find a^N-1 (mod N), mod_exp is O(n^3) speed
            return 'composite'  # if mod_exp returns something other than 1, it is composite
    return 'prime'  # if all tests pass, we say N is prime


# Time complexity is O(kn^3), space complexity if O(k) because we have k random numbers
def run_miller_rabin(N,k):
    rands = random.sample(range(1, N - 1), k)  # get k random numbers from a to N-1, random.sample is O(n) speed
    for a in rands:  # loop through the k random numbers
        exp = N-1  # exponent starts at N-1
        while True:
            b = mod_exp(a, exp, N)  # find a^exp (mod N), mod_exp is O(n^3) speed
            if b == N-1:  # N-1 (mod N) is equivalent to -1 (mod N), the test passes in this case
                break
            elif b != 1:  # if it isn't 1 or -1, then the test failed
                return 'composite'
            if exp % 2 == 1:  # once the exponent is odd, we break
                break
            exp = exp/2  # divide the exponent by 2
    return 'prime'  # if all tests pass, we say N is prime
