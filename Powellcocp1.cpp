//优化都不错，但是采样量比较高。N=20,Rosenbrock为5664，N=200为83622，2/3轮outer收敛（这个有点奇怪），且精度都很好；Rastrigin N=20 时为N=20为5540，N=200为64973,精度好；Schwefel N=20为5621，N=200为54941，精度略差；Logpotential N=20为5527，N=200为82928，精度好。可见基本线性增长。
//以上为DELTA=0.02/0.05，降低到0.01/0.025时，采样量相应翻倍：N=500 Logpotential 采样量从203600到324400。

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <unordered_map>

using namespace std;

// ---------------------- DIM & DOMAIN ----------------------
#define N 200

// 0:Rosenbrock, 1:Rastrigin, 2:Schwefel, 3:Ackley, 4:LogPotential
#define OBJTYPE 0

#if OBJTYPE == 0
static const double K = 2.0;
#elif OBJTYPE == 1
static const double K = 5.12;
#elif OBJTYPE == 2
static const double K = 500.0;
#elif OBJTYPE == 3
static const double K = 5.0;
#else
static const double K = 2.0;
#endif

static const double POWERN = 3.0;
static const double D = 0.02; //Linesearch精度

using Vec = array<double, N>;
int cnt = 0;
int clip = 0;
int cache_hits = 0;

// ---------------------- basic helpers ----------------------
static inline double dotv(const Vec& a, const Vec& b) {
    double s = 0;
    for (int i = 0; i < N; i++) s += a[i] * b[i];
    return s;
}

static inline double normv(const Vec& a) {
    return sqrt(dotv(a, a));
}

static inline Vec addv(const Vec& a, const Vec& b, double sb = 1.0) {
    Vec r;
    for (int i = 0; i < N; i++) r[i] = a[i] + sb * b[i];
    return r;
}

static inline Vec subv(const Vec& a, const Vec& b) {
    Vec r;
    for (int i = 0; i < N; i++) r[i] = a[i] - b[i];
    return r;
}

static inline Vec scalev(const Vec& a, double s) {
    Vec r;
    for (int i = 0; i < N; i++) r[i] = a[i] * s;
    return r;
}

static inline Vec normalize(Vec v) {
    double n = normv(v);
    if (n < 1e-18) return v;
    for (int i = 0; i < N; i++) v[i] /= n;
    return v;
}

static inline Vec normalize_inf(Vec v) {
    double m = 0.0;
    for (int i = 0; i < N; i++) m = max(m, fabs(v[i]));
    if (m < 1e-18) return v;
    for (int i = 0; i < N; i++) v[i] /= m;
    return v;
}

// static inline void clipdomain(Vec& x) {
//       for(int i=0; i<N; i++) {
//           if(x[i] > K) x[i] = K;
//           if(x[i] < -K) x[i] = -K;
//       }
//   }

static inline double sgn(double x) {
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    return 0.0;
}

// ---------------------- Line Cache ----------------------
// 对每条线 (x0, vunit) 缓存计算过的 s -> f 值
static inline long long qkey(double x) {
    // 量化到 1e-10，与原 operator== 的容差一致
    return llround(x * 1e10);
}

struct LineKey {
    Vec x0;
    Vec vunit;

    bool operator==(const LineKey& other) const {
        for (int i = 0; i < N; i++) {
            if (qkey(x0[i]) != qkey(other.x0[i])) return false;
            if (qkey(vunit[i]) != qkey(other.vunit[i])) return false;
        }
        return true;
    }
};

struct LineKeyHash {
    size_t operator()(const LineKey& k) const {
        size_t h = 0;
        for (int i = 0; i < N; i++) {
            long long a = qkey(k.x0[i]);
            long long b = qkey(k.vunit[i]);
            h ^= std::hash<long long>{}(a)+0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<long long>{}(b)+0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// 全局缓存：line -> (s -> fvalue)
unordered_map<LineKey, unordered_map<double, double>, LineKeyHash> line_cache;

static inline int quantize_s(double s, double grid_h) {
    return (int)round(s / grid_h);
}

// ---------------------- objective ----------------------
static inline double baseobjective(const Vec& x) {
    for (int i = 0; i < N; i++) {
        if (x[i] < -K || x[i] > K) { clip++; return 0.0; }
    }

#if OBJTYPE == 0
    // Inverted Rosenbrock (pairwise)
    double rosen = 0.0;
    for (int i = 0; i < N - 1; i++) {
        double x0 = x[i];
        double x1 = x[i + 1];
        double t1 = x1 - x0 * x0;
        double t2 = 1.0 - x0;
        rosen += 100.0 * t1 * t1 + t2 * t2;
    }
    double height = 60.0 * N - rosen;
    return (height > 0.0) ? height : 0.0;

#elif OBJTYPE == 1
    // Inverted Rastrigin
    const double A = 10.0;
    double val = A * N;
    for (int i = 0; i < N; i++) {
        val += x[i] * x[i] - A * cos(2.0 * M_PI * x[i]);
    }
    double height = 40.0 * N - val;
    return (height > 0.0) ? height : 0.0;

#elif OBJTYPE == 2
    // Inverted Schwefel
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double xi = x[i];
        s += -xi * sin(sqrt(fabs(xi)));
    }
    double height = 2000.0 - s;
    return (height > 0.0) ? height : 0.0;

#elif OBJTYPE == 3
    // Inverted Ackley
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < N; i++) {
        sum1 += x[i] * x[i];
        sum2 += cos(2.0 * M_PI * x[i]);
    }
    double a = 20.0, b = 0.2;
    double ack = -a * exp(-b * sqrt(sum1 / N)) - exp(sum2 / N) + a + exp(1.0);
    double height = 50.0 - ack;
    return (height > 0.0) ? height : 0.0;

#else
    // LogPotential
    double sumlog = 0.0;
    for (int i = 0; i < N; i++) {
        double xi = x[i];
        double t1 = xi - 0.5;
        double t2 = xi + 0.5;
        sumlog += log(t1 * t1 + 0.0001) + log(t2 * t2 + 0.01);
    }
    double C = 1.5;
    double height = C - sumlog;
    return (height > 0.0) ? height : 0.0;
#endif
}

static inline double fvalue(const Vec& x) {
    cnt++;
    double b = baseobjective(x);
    return (b > 0.0) ? pow(b, POWERN) : 0.0;
}

static inline double bvaluefromf(double fx) {
    return (fx > 0.0) ? pow(fx, 1.0 / POWERN) : 0.0;
}

static inline double bvalue(const Vec& x) {
    return bvaluefromf(fvalue(x));
}

// 带缓存的 fline
static inline double fline_cached(const Vec& x0, const Vec& vunit, double s,
    LineKey& key, unordered_map<double, double>& cache) {
    auto it = cache.find(s);
    if (it != cache.end()) {
        cache_hits++;
        return it->second;
    }

    Vec x = addv(x0, vunit, s);
    //clipdomain(x);
    //cnt++;
    double fval = fvalue(x);
    cache[s] = fval;
    return fval;
}

// ---------------------- COCP1 优化版 ----------------------
static const int COCP1_MAXITERS = 200;
static const int COCP1_PATIENCE = 5;
static const double COCP1_EPSINT = 1e-4;

// 统一网格 COCP1，严格按 (22) 增量维护
static Vec LineSearchCOCP1_Optimized(const Vec& x0, const Vec& vunit, double DELTA) {
    // 网格参数
    //const double h = DELTA;  // 统一网格步长
    //const double h_fine = DELTA / 10.0;  // 窗口内细分
    const int WINDOW_FINE = 5;  // 窗口细分数
    //const int TAIL_RADIUS = 200;  // 粗网格半径（单位：h）
    //改为动态TAIL_RADIUS
    double smin = -1e100, smax = 1e100;

    for (int i = 0; i < N; i++) {
        double vi = vunit[i];
        if (fabs(vi) < 1e-18) continue;
        double lo = (-K - x0[i]) / vi;
        double hi = (K - x0[i]) / vi;
        if (lo > hi) std::swap(lo, hi);
        smin = std::max(smin, lo);
        smax = std::min(smax, hi);
    }

    // 将尾部半径限制在可行区间内（再留一点 buffer）
    int TAIL_RADIUS = (int)ceil(std::max(fabs(smin), fabs(smax)) / DELTA) + 2;
    TAIL_RADIUS = std::min(TAIL_RADIUS, 100);

    // 构造 LineKey
    LineKey key;
    key.x0 = x0;
    key.vunit = vunit;

    // 获取或创建该线的缓存
    auto& cache = line_cache[key];

    // 当前位置
    int s_idx = 0;  // s = s_idx * h
    double bests = 0.0;
    double bestval = fline_cached(x0, vunit, 0.0, key, cache);

    auto upd_best = [&](double fv, double s) {
        if (fv > bestval) { bestval = fv; bests = s; }
        };

    // 初始化积分（按网格离散化）
    // Int1: 粗网格部分 sum_{i=1}^{TAIL} h * [f^N(s-ih) - f^N(s+ih)]
    double Int1 = 0.0;
    for (int i = 1; i <= TAIL_RADIUS; i++) {
        double s_neg = -i * DELTA;
        double s_pos = i * DELTA;
        double fm = fline_cached(x0, vunit, s_neg, key, cache);
        double fp = fline_cached(x0, vunit, s_pos, key, cache);

        Int1 += DELTA * (fm - fp);

        // ✅关键：这些点你已经算了，不要浪费
        upd_best(fm, s_neg);
        upd_best(fp, s_pos);
    }

    // Int2: 窗口细分部分 sum_{j=-W}^{W} (j*h_fine) * f^N(s + j*h_fine) * h_fine
    double Int2 = 0.0;
    for (int j = -WINDOW_FINE; j <= WINDOW_FINE; j++) {
        double s_win = j * DELTA / 10.0;
        double fval = fline_cached(x0, vunit, s_win, key, cache);
        Int2 += (j * DELTA / 10.0) * fval * 0.1;

        upd_best(fval, s_win);
    }

    double Int = Int1 - Int2;

    int noimprove = 0;
    int prev_s_idx = 0;

    for (int iter = 0; iter < COCP1_MAXITERS; iter++) {
        if (fabs(Int) < COCP1_EPSINT) break;

        // 根据迭代进度调整采样分辨率
        double resolution_factor = 1.0;
        if (iter > 20) resolution_factor = 0.5;
        if (iter > 50) resolution_factor = 0.25;
        if (iter > 100) resolution_factor = 0.1;

        // 应用到线搜索参数
        double h = DELTA * resolution_factor;
        double h_fine = DELTA / 10.0 * resolution_factor;

        double prevbest = bestval;
        prev_s_idx = s_idx;

        // 按梯度方向移动一个网格步
        int step_dir = (Int > 0) ? -1 : 1;  // Int>0 说明负方向梯度大，往负走
        s_idx += step_dir;
        double s = s_idx * h;

        double curval = fline_cached(x0, vunit, s, key, cache);
        if (curval > bestval) {
            bestval = curval;
            bests = s;
        }

        if (bestval > prevbest) {
            noimprove = 0;
        }
        else {
            noimprove++;
            //if(noimprove > COCP1_PATIENCE) break;
          //double Int_scale = bestval * h * TAIL_RADIUS;     // 一个很粗的量级估计
          //double Int_stop  = 1e-12 * Int_scale;             // 你可先用 1e-12~1e-10 试

            double Int_scale = fabs(Int1) + fabs(Int2) + 1.0;
            double Int_stop = 1e-3 * Int_scale;   // 先用 1e-10；若还早停可调到 1e-12
            if (noimprove > COCP1_PATIENCE || fabs(Int) < Int_stop) break;
        }

        // ====== 关键：按 (22) 增量更新 Int1 和 Int2 ======
        // (22): F(α2) - F(α1) = 
        //   Σ_{α2≤t<α1} [f^N(t) - f^N(-t)] * h (if α2 < α1, step=1)
        //   - window部分的增量

        double s_new = s_idx * h;
        double s_old = prev_s_idx * h;

        // Int1 增量：新覆盖的粗网格区间
        double Int1_delta = 0.0;
        if (step_dir > 0) {
            // 向右移动：新增 [s_old+TAIL*h+h, s_new+TAIL*h] 和 [s_new-TAIL*h, s_old-TAIL*h-h]
            for (int i = TAIL_RADIUS; i <= TAIL_RADIUS; i++) {
                double s_left_new = s_new - i * h;
                double s_left_old = s_old - i * h;
                double s_right_new = s_new + i * h;
                double s_right_old = s_old + i * h;

                // 新增左侧点
                if (s_left_new < s_left_old - h / 2) {
                    double fval = fline_cached(x0, vunit, s_left_new, key, cache);
                    Int1_delta += h * fval;
                }
                // 移除左侧旧点
                if (s_left_old < s_left_new - h / 2) {
                    double fval = fline_cached(x0, vunit, s_left_old, key, cache);
                    Int1_delta -= h * fval;
                }
                // 新增右侧点
                if (s_right_new > s_right_old + h / 2) {
                    double fval = fline_cached(x0, vunit, s_right_new, key, cache);
                    Int1_delta -= h * fval;
                }
                // 移除右侧旧点
                if (s_right_old > s_right_new + h / 2) {
                    double fval = fline_cached(x0, vunit, s_right_old, key, cache);
                    Int1_delta += h * fval;
                }
            }
        }
        else {
            // 向左移动：类似处理
            for (int i = TAIL_RADIUS; i <= TAIL_RADIUS; i++) {
                double s_left_new = s_new - i * h;
                double s_left_old = s_old - i * h;
                double s_right_new = s_new + i * h;
                double s_right_old = s_old + i * h;

                if (s_left_new < s_left_old - h / 2) {
                    double fval = fline_cached(x0, vunit, s_left_new, key, cache);
                    Int1_delta += h * fval;
                }
                if (s_left_old < s_left_new - h / 2) {
                    double fval = fline_cached(x0, vunit, s_left_old, key, cache);
                    Int1_delta -= h * fval;
                }
                if (s_right_new > s_right_old + h / 2) {
                    double fval = fline_cached(x0, vunit, s_right_new, key, cache);
                    Int1_delta -= h * fval;
                }
                if (s_right_old > s_right_new + h / 2) {
                    double fval = fline_cached(x0, vunit, s_right_old, key, cache);
                    Int1_delta += h * fval;
                }
            }
        }

        Int1 += Int1_delta;

        // Int2 重新计算（窗口较小，直接重算）
        double Int2_new = 0.0;
        for (int j = -WINDOW_FINE; j <= WINDOW_FINE; j++) {
            double s_win = s_new + j * h_fine;
            double fval = fline_cached(x0, vunit, s_win, key, cache);
            Int2_new += (j * h_fine) * fval * h_fine;
        }

        Int = Int1 - Int2_new;
        Int2 = Int2_new;
    }

    Vec x1 = addv(x0, vunit, bests);
    //clipdomain(x1);
    return x1;
}

// ---------------------- Classical Powell's Method ----------------------
static const int MAXOUTER = 50;

static void printvec(const Vec& v) {
    cout << "[";
    for (int i = 0; i < N; i++) {
        cout << v[i];
        if (i + 1 < N) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout.setf(std::ios::fixed);
    cout << setprecision(6);

    // Initialize starting point
    Vec x;
    for (int i = 0; i < N; i++) x[i] = -0.5 + 0.0001 * i;
    //clipdomain(x);

    double bestF = bvalue(x);
    Vec bestX = x;

    // Initialize direction set to coordinate axes
    vector<Vec> dirs(N);
    for (int i = 0; i < N; i++) {
        dirs[i].fill(0.0);
        dirs[i][i] = 1.0;
    }

    cout << "Init: objType=" << OBJTYPE << ", N=" << N
        << ", best=" << bestF << endl;

    for (int outer = 0; outer < MAXOUTER; outer++) {
        // Delta schedule: coarse -> fine
        double DELTA = (outer < MAXOUTER / 10) ? 2.5 * D * K / 2.0 : D * K / 2.0;

        Vec x0 = x;
        double f0 = bvalue(x);

        int max_increase_idx = -1;
        double max_increase = 0.0;

        // Line search along each direction
        for (int i = 0; i < N; i++) {
            double fprev = bestF;

            Vec dir_normalized = normalize_inf(dirs[i]);

            // 使用优化版 COCP1
            Vec xnew = LineSearchCOCP1_Optimized(x, dir_normalized, DELTA);
            double fnew = bvalue(xnew);

            double increase = fnew - fprev;
            if (increase > max_increase) {
                max_increase = increase;
                max_increase_idx = i;
            }

            x = xnew;

            if (fnew > bestF) {
                bestF = fnew;
                bestX = x;
            }
        }

        // Powell direction update
        Vec xN = x;
        double fN = bvalue(xN);

        Vec d = subv(xN, x0);
        double dnorm = normv(d);

        if (dnorm > 1e-8) {
            //Vec d_normalized = normalize(d);
            Vec d_normalized = normalize_inf(d);
            Vec x_extra = LineSearchCOCP1_Optimized(xN, d_normalized, DELTA);
            double f_extra = bvalue(x_extra);

            // 替换贡献最大的方向
            if (max_increase_idx >= 0) {
                dirs[max_increase_idx] = d;
            }
            else {
                for (int i = 0; i < N - 1; i++) {
                    dirs[i] = dirs[i + 1];
                }
                dirs[N - 1] = d;
            }

            if (f_extra > fN) {
                x = x_extra;
                if (f_extra > bestF) {
                    bestF = f_extra;
                    bestX = x;
                }
            }
        }

        double gain = 0;
        if (bestF > f0) { gain = bestF - f0; x=bestX;}
        else {x=x0;}//= bvalue(x) - f0;

        if (outer % 1 == 0 || outer == MAXOUTER - 1) {
            cout << "Outer " << outer
                << ", best=" << bestF
                << ", gain=" << gain
                << ", samples=" << cnt - clip
                << ", cache_hits=" << cache_hits
                << ", hit_rate=" << (cnt > 0 ? 100.0 * cache_hits / (cnt - clip + cache_hits) : 0) << "%"
                << endl;
        }

        if (gain < 1e-8) {
            cout << "Converged at outer " << outer << endl;
            break;
        }
    }

    cout << "\nDONE" << endl;
    cout << "Best: " << bestF << endl;
    cout << "Best x: ";
    printvec(bestX);
    cout << endl;
    cout << "Total samples: " << cnt - clip << ", clip=" << clip << endl;
    cout << "Cache hits: " << cache_hits << " ("
        << (cnt > 0 ? 100.0 * cache_hits / (cnt - clip + cache_hits) : 0) << "%)" << endl;

    return 0;
}
