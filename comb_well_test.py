import random
import sys

# random.seed(42)

if __name__ == "__main__":
    rounds = int(sys.argv[1])
    bp_count = int(sys.argv[2])

    MUTS = "ABCDEFGHIJKLMNO"[:bp_count]

    INCLUSION_ODDS = {m: random.random() for m in MUTS}
    for bp, odds in INCLUSION_ODDS.items():
        print(bp, odds)

    OVERLAP_MAP = {m: [] for m in MUTS}
    # OVERLAP_MAP["A"] = ["B"]
    # OVERLAP_MAP["B"] = ["A"]

    def sort_muts(mut, new_bp) -> str:
        chars = set(mut)

        if random.random() < INCLUSION_ODDS[new_bp]:
            chars.add(new_bp)
        else:
            return ""

        for overlap in OVERLAP_MAP[new_bp]:
            chars.discard(overlap)

        return "".join(sorted(list(chars)))

    # map from mutation string to its estimated count
    mutations = {
        m: 1 if random.random() < INCLUSION_ODDS[m] else 0 for m in MUTS
    }  # starting after first round

    for _ in range(rounds - 1):  # 3 more rounds
        for m in list(mutations.keys()):
            for bp in [bp for bp in MUTS]:  # 8 PCR primer pairs
                new_mut = sort_muts(m, bp)

                if not new_mut or bp not in new_mut:
                    continue

                if new_mut in mutations:
                    mutations[new_mut] += 1
                else:
                    mutations[new_mut] = 1

    # for m, count in mutations.items():
    #     print(m, count)

    # count the inclusion ratio of A
    for test_m in MUTS:
        a_count = 0
        total_count = 0
        for m in mutations:
            mut_count = mutations[m]
            if test_m in m:
                a_count += mut_count
            total_count += mut_count
        print(str(float(a_count) / float(total_count)))
