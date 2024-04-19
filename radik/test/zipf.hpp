/**
 * Reference:
 * https://jasoncrease.medium.com/rejection-sampling-the-zipf-distribution-6b359792cffa
 */
#include <cmath>
#include <random>

template <class RNG>
class ZipfRejectionSampler {
public:
    ZipfRejectionSampler(RNG random_generator, long N, double skew)
        : _rand(random_generator), _N(N), _skew(skew),
        _t((pow(N, 1 - skew) - skew) / (1 - skew)) {}

    long getSample() {
        while (true) {
            double invB = bInvCdf(nextDouble());
            long sampleX = (long)(invB + 1);
            double yRand = nextDouble();
            double ratioTop = pow(sampleX, -_skew);
            double ratioBottom = sampleX <= 1 ? 1  / _t : pow(invB, -_skew)  / _t;
            double rat = (ratioTop) / (ratioBottom * _t);

            if (yRand < rat)
                return sampleX;
        }
    }

private:
    double bInvCdf(double p) {
        if (p * _t <= 1)
            return p * _t;
        else
            return pow((p * _t) * (1 - _skew) + _skew, 1 / (1 - _skew) );
    }

    double nextDouble() {
        std::uniform_real_distribution<double> dis(0.f, 1.f);
        return dis(_rand);
    }

    RNG _rand;
    const long _N;
    const double _skew;
    const double _t;
};
