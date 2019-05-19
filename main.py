import os, sys

def main(repeat):
    use_shakeshake = 0
    for repeat in range(repeat):
        for mnist in [0, 1]:
            for act_type in ["relu", "regact", "linkact"]:
                os.system(
                    "python3 train.py --mnist %d --act_type %s --use_shakeshake %d" % (
                        mnist, act_type, use_shakeshake
                    )
                )

if __name__ == "__main__":

    repeat = int(sys.argv[1])
    main(repeat)
