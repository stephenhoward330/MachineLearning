import os
from passlib.hash import md5_crypt


def gen_passwd(directory, usernames):
    path = os.path.join(directory, "passwd")

    ctr = 0
    with open(path, 'w') as f:
        for username in usernames:
            f.write(username + ":x:" + str(1000+ctr) + ":" + str(1010+ctr) + ":DEREK:directory:shell" + str(10+ctr) + '\n')
            ctr += 1


def gen_shadow(directory, usernames, passwords):
    path = os.path.join(directory, "shadow")

    with open(path, 'w') as f:
        for i in range(len(usernames)):
            hashed = md5_crypt.hash(passwords[i])
            f.write(usernames[i] + ":" + hashed + ":12345:0:99999:7:::\n")


if __name__ == '__main__':
    users = ["user2", "user3"]
    passes = ["UpPeRiNwEiRdPlAcEs", "PassingAll11!@"]

    gen_passwd("verystrong/etc", users)
    gen_shadow("verystrong/etc", users, passes)
