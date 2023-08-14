---
title: "Sentence Similarity"
url: /projects/sentences
type: posts
desc: [
    "<b>code</b>: <a href=\"https://github.com/kali-v/sentence-similarity\"> here </a>"
]
---

Hashing table-based solution for the sentence similarity problem, achieving near-linear time complexity - considering 0 and 1 distances (distance-1 defined as "addition or deletion of a single word" with input from dist-0)

# Distance 0

Hash is computed using \\[ (31 · hash + 7 · c)\ mod\ cap \\], while iterating over the sentence
- `hash` is the current value of the hash
- `c` is the current character in the sentence
- the `cap` is a big number limiting overflow.

The sentence is sent to output only if a computed hash appears for the first time.

```cpp
int Dist0HashTable::hash(std::string sentence, int cap) {
    ulong long hash = 1;
    for (char c : sentence) {
        hash = (31 * hash + 7 * c) % cap;
    }
    return hash;
};

int Dist0HashTable::insert(char* value) {
    int hash_val = this->hash(value, capacity);
    if (this->table[hash_val] == nullptr) {
        this->table[hash_val] = value;
        return 0;
    }
    return 1;
}
```

![dist0](/images/sentences/dist0.png)

# Distance 1

When processing a sentence, we stored several pieces of information: the number of words,
the number of characters, and the sum of the character values in the first and second parts of the
sentence. We know that distance-1 sentences can only be obtained by adding or removing
a single word. Therefore, to find distance-1 sentences for any given sentence, we will focus on
sentences that have one more or one fewer word than the original sentence. We also want to inspect
only sentences that have just a few more or fewer characters that should correspond with the change
of adding/removing a single word. With this information, we can construct our hash table. The first
level of the table will be organized based on the number of words in each sentence. In the second
level, the sentences will be divided into buckets according to: the number of chars / 5. This way, we can
quickly filter a lot of sentences. After identifying buckets in a hash table, we iterate
over a list of hashes, and we declare sentences are distance-1 iff the sum of chars is equal for one-half
of the sentences and the difference in sums of characters in the whole sentence differs by <600.

```cpp
int Dist1HashTable::insert(std::string value) {
    int sval = 0;
    int wc = 0;
    int cc = 0;
    int dist = 0;
    int split_it = 2;
    int w_sens = 600; // max allowable diff in sums of chars

    for (char cr : value) {
        if (cr == ' ') wc++;
        if (cc > (int)(value.length() / split_it)) {
            dist = sval;
            sval = 0;
            split_it--;
        }
        cc++;
        sval += cr;
    }

    std::vector<int> hash{dist + sval, dist, sval};

    cc /= 5;
    cc = cc < ccm - 1 ? cc : ccm - 2;
    wc = wc < wcm - 1 ? wc : wcm - 2;

    int f = 0;
    if (wc > 0) {
#pragma omp parallel for
        for (int i = 0; i < this->table[wc - 1][cc].size(); i++) {
            if (((hash[1] == this->table[wc - 1][cc][i][1]) != (hash[2] == this->table[wc - 1][cc][i][2]))) {
                if (abs(hash[0] - this->table[wc - 1][cc][i][0]) < w_sens) {
                    f = 1;
                    i = this->table[wc - 1][cc].size();
                }
            }
        }
    }

    if (f == 0) {
#pragma omp parallel for
        for (int i = 0; i < this->table[wc + 1][cc].size(); i++) {
            if (((hash[1] == this->table[wc + 1][cc][i][1]) != (hash[2] == this->table[wc + 1][cc][i][2]))) {
                if (abs(hash[0] - this->table[wc + 1][cc][i][0]) < w_sens) {
                    f = 1;
                    i = this->table[wc + 1][cc].size();
                }
            }
        }
    }

    if (f == 1) return 1;

    this->table[wc][cc].push_back(hash);

    return 0;
}
```


Distance-1 algorithm on my setup(12th Gen Intel(R) Core(TM) i5-12600H, 16GB RAM) was
able to achieve near linear execution time. However, to achieve that, it uses multi-threading in a
way that makes the algorithm non-deterministic (some lines might be added to the hash-table before
lines that are actually first in the input file). Thus, the output lines will be slightly different every
time and will also cause different wallclock times.


![dist1](/images/sentences/dist1.png)



