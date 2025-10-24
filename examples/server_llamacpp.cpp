// Example integration snippet for llama.cpp
#include <iostream>
#include <vector>

// Pseudo-interfaces demonstrating where kvseed-guard hooks in.
struct KvSeed {
    std::vector<float> key;
    std::vector<float> value;
    std::vector<int> token_ids;
};

struct RuntimeContext {
    KvSeed kv_seed;
};

void apply_seed(RuntimeContext &ctx, const KvSeed &seed) {
    ctx.kv_seed = seed;
}

int main() {
    RuntimeContext ctx;
    KvSeed seed;
    // Load seed tensors via Python tooling and feed them here.
    seed.token_ids = {101, 102, 103, 104};
    seed.key.assign(32, 0.1f);
    seed.value.assign(32, 0.2f);
    apply_seed(ctx, seed);
    std::cout << "Seed injected; llama.cpp loop should consume ctx.kv_seed before decode" << std::endl;
    return 0;
}
