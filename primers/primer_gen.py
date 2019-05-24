import random

import primer3


def make_primers(n=200000):
    """Create a bunch of random sequences between 13 and 20 bp."""

    bp = "ATGC"

    seq_to_tm = []  # map from sequence to estimated tm
    for _ in range(n):
        primer_len = random.randint(10, 30)
        primer_seq = "".join([bp[random.randint(0, 3)] for _ in range(primer_len)])
        primer_tm = primer3.calcTm(primer_seq)
        primer_hairpin = primer3.calcHairpin(primer_seq).dg

        seq_to_tm.append((primer_seq, primer_tm, primer_hairpin))

    with open("primers.csv", "w") as output:
        output.write("seq,tm,hairpin\n")
        for seq, tm, hairpin in seq_to_tm:
            output.write(f"{seq},{tm},{hairpin}\n")


if __name__ == "__main__":
    make_primers()
