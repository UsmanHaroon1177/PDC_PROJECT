// src/main.cpp
#include <mpi.h>
#include <omp.h>
#include <metis.h>

#include <bits/stdc++.h>
using namespace std;
static constexpr double INF = numeric_limits<double>::infinity();

// ——— Graph & Edge definitions ————————————————————————————————

using WeightVec = vector<double>;
struct Edge { int to; WeightVec w; };
struct Graph {
    int n;
    vector<vector<Edge>> adj;
    Graph(int N = 0) : n(N), adj(N) {}
    void add_edge(int u, int v, const WeightVec& w) {
        adj[u].push_back({ v,w });
    }
};

// ——— 1) Read MTX (multi-objective) —————————————————————————————

Graph read_mtx(const string& fn, int& k_out) {
    ifstream in(fn);
    if (!in) throw runtime_error("Cannot open MTX: " + fn);
    string line;
    // skip header/comments
    while (getline(in, line)) {
        auto p = line.find_first_not_of(" \t\r\n");
        if (p == string::npos) continue;
        if (line[p] == '%' || line[p] == '#') continue;
        break;
    }
    if (line.empty()) throw runtime_error("MTX header missing");
    istringstream hh(line);
    int R, C, M; hh >> R >> C >> M;
    Graph G(R);
    bool saw = false; int k = 0;
    while (getline(in, line)) {
        auto p = line.find_first_not_of(" \t\r\n");
        if (p == string::npos || line[p] == '%' || line[p] == '#') continue;
        istringstream ss(line);
        vector<string> tok{ istream_iterator<string>(ss),{} };
        if (tok.size() < 2) continue;
        int u = stoi(tok[0]) - 1, v = stoi(tok[1]) - 1;
        if (!saw) { k = max(1, (int)tok.size() - 2); k_out = k; saw = true; }
        WeightVec w(k, 1.0);
        for (int i = 0; i < k && i + 2 < (int)tok.size(); i++)
            w[i] = stod(tok[i + 2]);
        if (u < 0 || u >= R || v < 0 || v >= R) throw runtime_error("Edge out of range");
        G.add_edge(u, v, w);
        G.add_edge(v, u, w);
    }
    if (!saw) k_out = 1;
    return G;
}

// ——— 2) METIS partition on rank 0 (then broadcast) ———————————————

void metis_partition_root(const Graph& G, int nparts, vector<int>& part, idx_t& edgecut, bool allow_nc) {
    idx_t nv = G.n, ncon = 1, objval;
    vector<idx_t> xadj(nv + 1), adjncy;
    adjncy.reserve(nv * 4);
    idx_t ec = 0;
    for (int i = 0; i < nv; i++) {
        xadj[i] = ec;
        for (auto& e : G.adj[i]) { adjncy.push_back(e.to); ec++; }
    }
    xadj[nv] = ec;
    part.assign(nv, 0);
    vector<real_t> tpwgts(nparts, 1.0 / (double)nparts), ubvec(1, 1.05);
    idx_t options[METIS_NOPTIONS]; METIS_SetDefaultOptions(options);
    options[METIS_OPTION_CONTIG] = allow_nc ? 0 : 1;
    options[METIS_OPTION_NUMBERING] = 0;
    int status = METIS_PartGraphKway(
        &nv, &ncon,
        xadj.data(), adjncy.data(),
        nullptr, nullptr, nullptr,
        (idx_t*)&nparts,
        tpwgts.data(), ubvec.data(),
        options,
        &objval,
        (idx_t*)part.data()
    );
    if (status != METIS_OK) { cerr << "[ERROR] METIS failed\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
    edgecut = objval;
}

// ——— 3) extract local subgraph for this rank —————————————————————————

void extract_local(const Graph& G, const vector<int>& part, int rank,
    Graph& localG, vector<int>& g2l, vector<int>& l2g) {
    int N = G.n;
    g2l.assign(N, -1);
    for (int i = 0, c = 0; i < N; i++) {
        if (part[i] == rank) { g2l[i] = c++; l2g.push_back(i); }
    }
    localG = Graph(l2g.size());
    for (int u_loc = 0; u_loc < (int)l2g.size(); u_loc++) {
        int gu = l2g[u_loc];
        for (auto& e : G.adj[gu]) {
            if (part[e.to] == rank) localG.add_edge(u_loc, g2l[e.to], e.w);
        }
    }
}

// ——— 4) Distributed Bellman–Ford (initial SOSP) ——————————————————————

void distributed_bf(const Graph& G, const Graph& localG,
    const vector<int>& l2g, const vector<int>& part, const vector<int>& g2l,
    int src_global, vector<double>& dist, vector<int>& par)
{
    int world_size, rank; MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nloc = localG.n;
    dist.assign(nloc, INF);
    par.assign(nloc, -1);
    if (src_global >= 0 && src_global < (int)g2l.size()) {
        int ls = g2l[src_global];
        if (ls >= 0) { dist[ls] = 0.0; par[ls] = -1; }
    }
    // gather cross-partition edges
    vector<tuple<int, int, double>> cross;
    for (int u_loc = 0; u_loc < nloc; u_loc++) {
        int gu = l2g[u_loc];
        for (auto& e : G.adj[gu]) {
            if (part[e.to] != rank) cross.emplace_back(u_loc, e.to, e.w[0]);
        }
    }
    int global_changed = 1;
    while (global_changed) {
        int local_changed = 0;
        // relax within localG
#pragma omp parallel for reduction(|:local_changed) schedule(dynamic)
        for (int u = 0; u < nloc; u++) {
            double du = dist[u];
            if (du == INF) continue;
            for (auto& e : localG.adj[u]) {
                int v = e.to;
                double nd = du + e.w[0];
                if (nd < dist[v]) {
#pragma omp critical
                    if (nd < dist[v]) { dist[v] = nd; par[v] = u; local_changed = 1; }
                }
            }
        }
        // prepare cross updates
        vector<vector<pair<int, double>>> sbuf(world_size);
        for (auto& t : cross) {
            int u_loc, gv; double w;
            tie(u_loc, gv, w) = t;
            double du = dist[u_loc];
            if (du == INF) continue;
            sbuf[part[gv]].emplace_back(gv, du + w);
        }
        // flatten and all-to-all
        vector<int> scount(world_size), sdisp(world_size);
        for (int r = 0; r < world_size; r++) scount[r] = sbuf[r].size();
        sdisp[0] = 0;
        for (int r = 1; r < world_size; r++) sdisp[r] = sdisp[r - 1] + scount[r - 1];
        int send_tot = sdisp.back() + scount.back();
        vector<int> sidx(send_tot);
        vector<double> sdist(send_tot);
        for (int r = 0; r < world_size; r++) {
            int off = sdisp[r];
            for (int i = 0; i < scount[r]; i++) {
                sidx[off + i] = sbuf[r][i].first;
                sdist[off + i] = sbuf[r][i].second;
            }
        }
        vector<int> rcount(world_size), rdisp(world_size);
        MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
        rdisp[0] = 0;
        for (int r = 1; r < world_size; r++) rdisp[r] = rdisp[r - 1] + rcount[r - 1];
        int recv_tot = rdisp.back() + rcount.back();
        vector<int> ridx(recv_tot);
        vector<double> rdist(recv_tot);
        MPI_Alltoallv(sidx.data(), scount.data(), sdisp.data(), MPI_INT,
            ridx.data(), rcount.data(), rdisp.data(), MPI_INT,
            MPI_COMM_WORLD);
        MPI_Alltoallv(sdist.data(), scount.data(), sdisp.data(), MPI_DOUBLE,
            rdist.data(), rcount.data(), rdisp.data(), MPI_DOUBLE,
            MPI_COMM_WORLD);
        // unpack
        for (int i = 0; i < recv_tot; i++) {
            int gv = ridx[i];
            double nd = rdist[i];
            int lv = g2l[gv];
            if (lv >= 0 && nd < dist[lv]) {
#pragma omp critical
                if (nd < dist[lv]) { dist[lv] = nd; par[lv] = -1; local_changed = 1; }
            }
        }
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    }
}

// ——— 5) Gather & dump distances (+ parents for MOSP) ——————————————————

template<typename T>
void gather_field(const vector<int>& l2g, const vector<T>& loc, int N_glob, vector<T>& glob) {
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int nloc = l2g.size();
    vector<int> counts(size), displ(size);
    MPI_Gather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        displ[0] = 0;
        for (int r = 1; r < size; r++) displ[r] = displ[r - 1] + counts[r - 1];
        glob.assign(N_glob, T());
    }
    vector<int> all_idx(displ.back() + counts.back());
    vector<T> all_loc(all_idx.size());
    MPI_Gatherv(l2g.data(), nloc, MPI_INT,
        rank == 0 ? all_idx.data() : nullptr,
        counts.data(), displ.data(), MPI_INT,
        0, MPI_COMM_WORLD);
    MPI_Gatherv(loc.data(), nloc, (sizeof(T) == sizeof(double) ? MPI_DOUBLE : MPI_INT),
        rank == 0 ? (void*)all_loc.data() : nullptr,
        counts.data(), displ.data(), (sizeof(T) == sizeof(double) ? MPI_DOUBLE : MPI_INT),
        0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            for (int i = 0; i < counts[r]; i++) {
                int pos = displ[r] + i;
                glob[all_idx[pos]] = all_loc[pos];
            }
        }
    }
}

void gather_and_output(const vector<int>& l2g, const vector<double>& dist_local,
    int N_global, const string& out_csv)
{
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    vector<double> dist_glob;
    gather_field(l2g, dist_local, N_global, dist_glob);
    if (rank == 0) {
        cout << "[sample global distances]\n";
        for (int i = 0; i < min(25, N_global); i++) {
            double d = dist_glob[i];
            cout << setw(4) << i << " : " << (d == INF ? "inf" : to_string(d)) << "\n";
        }
        ofstream out(out_csv);
        out << "vertex,distance\n";
        for (int i = 0; i < N_global; i++) {
            double d = dist_glob[i];
            out << i << "," << (d == INF ? "inf" : to_string(d)) << "\n";
        }
        cout << "Full distances written to " << out_csv << "\n";
    }
}

// ——— 6) Shared-memory SOSP_Update (Algo 1) ——————————————————————————

struct UpdateStats { int affected = 0, iters = 0; };
void SOSP_Update(const Graph& G, vector<double>& dist,
    const vector<tuple<int, int, WeightVec>>& ins, int obj, UpdateStats& st)
{
    int n = G.n;
    vector<char> marked(n, 0);
    vector<int> affected;
    // group
    vector<vector<pair<int, double>>> I(n);
    for (auto& t : ins) {
        int u, v; WeightVec w;
        tie(u, v, w) = t;
        if (u >= 0 && u < n && v >= 0 && v < n) I[v].emplace_back(u, w[obj]);
    }
    // Step 1
#pragma omp parallel for schedule(dynamic)
    for (int v = 0; v < n; v++) {
        for (auto& p : I[v]) {
            double nd = dist[p.first] + p.second;
            if (nd < dist[v]) {
                dist[v] = nd;
#pragma omp critical
                if (!marked[v]) { marked[v] = 1; affected.push_back(v); }
            }
        }
    }
    st.affected = affected.size();
    // Step 2
    while (!affected.empty()) {
        st.iters++;
        vector<int> N;
        for (int v : affected)
            for (auto& e : G.adj[v]) N.push_back(e.to);
        sort(N.begin(), N.end());
        N.erase(unique(N.begin(), N.end()), N.end());
        affected.clear();
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)N.size(); i++) {
            int v = N[i];
            for (auto& e : G.adj[v]) {
                int u = e.to; double w = e.w[obj];
                if (marked[u]) {
                    double nd = dist[u] + w;
                    if (nd < dist[v]) {
                        dist[v] = nd;
#pragma omp critical
                        if (!marked[v]) { marked[v] = 1; affected.push_back(v); }
                    }
                }
            }
        }
    }
}

// ——— 7) MOSP heuristic (Algo 2) ————————————————————————————————

vector<double> mosp_heuristic(int N, int k, const vector<vector<int>>& par) {
    map<pair<int, int>, int> freq;
    for (int i = 0; i < k; i++) {
        for (int v = 0; v < N; v++) {
            int u = par[i][v];
            if (u >= 0) freq[{u, v}]++;
        }
    }
    Graph E(N);
    for (auto& kv : freq) {
        auto uv = kv.first;
        int x = kv.second;
        double w = double(k - x + 1);
        E.add_edge(uv.first, uv.second, WeightVec{ w });
    }
    vector<double> dist(N, INF);
    vector<char> vis(N, 0);
    struct Node { double d; int u; };
    auto cmp = [](Node const& a, Node const& b) {return a.d > b.d; };
    priority_queue<Node, vector<Node>, decltype(cmp)> pq(cmp);
    dist[0] = 0; pq.push({ 0,0 });
    while (!pq.empty()) {
        auto [du, u] = pq.top(); pq.pop();
        if (vis[u])continue;
        vis[u] = 1;
        for (auto& e : E.adj[u]) {
            double nd = du + e.w[0];
            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                pq.push({ nd,e.to });
            }
        }
    }
    return dist;
}

// ——— main() ——————————————————————————————————————————————

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (!rank) cerr << "Usage: " << argv[0] << " graph.mtx [--allow-noncontig] [-t threads]\n";
        MPI_Finalize(); return 1;
    }

    // parse args
    bool allow_nc = false;
    int user_threads = 0;
    for (int i = 2; i < argc; i++) {
        if (string(argv[i]) == "--allow-noncontig") allow_nc = true;
        if (string(argv[i]) == "-t" || string(argv[i]) == "--threads") {
            if (i + 1 < argc) user_threads = stoi(argv[++i]);
        }
    }
    if (user_threads > 0) omp_set_num_threads(user_threads);

    double t0 = MPI_Wtime();

    // 1) read
    int k; Graph G = read_mtx(argv[1], k);
    int N = G.n;
    double t_read = MPI_Wtime() - t0;

    // 2) partition
    vector<int> part; idx_t edgecut = 0;
    if (!rank) metis_partition_root(G, size, part, edgecut, allow_nc);
    MPI_Bcast(&edgecut, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    part.resize(N);
    MPI_Bcast(part.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
    double t_part = MPI_Wtime() - t0 - t_read;

    // 3) extract local
    Graph localG; vector<int> g2l, l2g;
    extract_local(G, part, rank, localG, g2l, l2g);
    if (l2g.empty()) {
        if (!rank) cerr << "[Rank " << rank << "] no local vertices; exiting\n";
        MPI_Finalize(); return 0;
    }

    // 4) initial distributed Bellman–Ford
    vector<double> dist_local;
    vector<int> par_local;
    distributed_bf(G, localG, l2g, part, g2l, 0, dist_local, par_local);
    double t_sssp = MPI_Wtime() - t0 - t_read - t_part;

    // 5) dump initial distances
    gather_and_output(l2g, dist_local, N, "distances.csv");

    // 6) simulate 100 insertions
    vector<tuple<int, int, WeightVec>> ins_glob;
    if (!rank) {
        int cnt = 0;
        for (int u = 0; u < N && cnt < 50000; u++) {
            for (auto& e : G.adj[u]) {
                ins_glob.emplace_back(u, e.to, e.w);
                if (++cnt >= 50000)break;
            }
        }
    }
    int nin = ins_glob.size();
    MPI_Bcast(&nin, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank) ins_glob.resize(nin);
    for (int i = 0; i < nin; i++) {
        int u, v; WeightVec w(k);
        if (!rank) tie(u, v, w) = ins_glob[i];
        MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(w.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank) ins_glob[i] = make_tuple(u, v, w);
    }
    // filter local
    vector<tuple<int, int, WeightVec>> ins_loc;
    for (auto& t : ins_glob) {
        int u, v; WeightVec w;
        tie(u, v, w) = t;
        if (u >= 0 && u < N && v >= 0 && v < N && g2l[u] >= 0 && g2l[v] >= 0)
            ins_loc.emplace_back(g2l[u], g2l[v], w);
    }

    // 7) dynamic update (Algorithm 1)
    UpdateStats stats;
    double t_upd0 = MPI_Wtime();
    for (int obj = 0; obj < k; obj++) {
        SOSP_Update(localG, dist_local, ins_loc, obj, stats);
    }
    double t_upd = MPI_Wtime() - t_upd0;

    // 8) dump updated distances
    gather_and_output(l2g, dist_local, N, "distances_after_update.csv");

    // 9) gather parents and run MOSP heuristic on rank 0
    vector<vector<int>> par_glob(k);
    vector<int> tmp_par;
    gather_field(l2g, par_local, N, tmp_par);
    if (!rank) {
        // replicate for each objective (demo only k=1)
        for (int i = 0; i < k; i++) par_glob[i] = tmp_par;
    }
    vector<double> final_dist;
    if (!rank) final_dist = mosp_heuristic(N, k, par_glob);

    // 10) final report
    if (!rank) {
        cout << "\n=== RESULTS ===\n";
        cout << "MPI ranks           : " << size << "\n";
        cout << "OpenMP threads      : " << (user_threads > 0 ? user_threads : omp_get_max_threads()) << "\n";
        cout << "METIS edgecut       : " << edgecut << "\n";
        cout << "Read time           : " << t_read << " s\n";
        cout << "Partition time      : " << t_part << " s\n";
        cout << "Init SSSP time      : " << t_sssp << " s\n";
        cout << "Update time         : " << t_upd << " s  "
            << " (affected=" << stats.affected << ", iters=" << stats.iters << ")\n";
        cout << "TOTAL runtime       : " << MPI_Wtime() - t0 << " s\n";
        cout << "[sample MOSP dist]\n";
        for (int i = 0; i < min(10, N); i++) {
            double d = i < final_dist.size() ? final_dist[i] : INF;
            cout << setw(4) << i << " : " << (d == INF ? "inf" : to_string(d)) << "\n";
        }
        cout << "==================\n";
    }

    MPI_Finalize();
    return 0;
}
