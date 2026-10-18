// Sorts each case of (value, tag) entries with libstdc++'s std::sort, or std::partial_sort(first, last, last),
// comparing by value only as XGBoost's Queue::MakeSummary does. Reads and writes raw int32 pairs.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct QEntry {
  float value;
  int32_t tag;
  bool operator<(const QEntry& b) const { return value < b.value; }
};

static std::vector<int32_t> killerVal;
static int32_t gas, nsolid, candidate;

static bool antiCmp(int32_t x, int32_t y) {
  if (killerVal[x] == gas && killerVal[y] == gas) {
    killerVal[x == candidate ? x : y] = nsolid++;
  }
  if (killerVal[x] == gas) candidate = x;
  else if (killerVal[y] == gas) candidate = y;
  return killerVal[x] < killerVal[y];
}

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "killer") == 0) {
    int32_t n = std::atoi(argv[2]);
    killerVal.assign(n, n);
    gas = n; nsolid = 0; candidate = 0;
    std::vector<int32_t> idx(n);
    for (int32_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), antiCmp);
    std::fwrite(killerVal.data(), sizeof(int32_t), n, stdout);
    return 0;
  }
  bool partial = argc > 1 && std::strcmp(argv[1], "partial") == 0;
  int64_t n;
  while (std::fread(&n, sizeof(n), 1, stdin) == 1) {
    std::vector<QEntry> q(n);
    for (int64_t i = 0; i < n; ++i) {
      int32_t bits, tag;
      if (std::fread(&bits, 4, 1, stdin) != 1 || std::fread(&tag, 4, 1, stdin) != 1) return 1;
      std::memcpy(&q[i].value, &bits, 4);
      q[i].tag = tag;
    }
    if (partial) std::partial_sort(q.begin(), q.end(), q.end());
    else std::sort(q.begin(), q.end());
    for (auto& e : q) {
      int32_t bits;
      std::memcpy(&bits, &e.value, 4);
      std::fwrite(&bits, 4, 1, stdout);
      std::fwrite(&e.tag, 4, 1, stdout);
    }
  }
  return 0;
}
