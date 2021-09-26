#include "../VectorCodec.hpp"
#include <cassert>
#include <vector>
#include <random>



int main()
{
    using namespace std;
    ranlux48 engine;
    for (int n = 16; n != 65536; n *= 2)
    {
        for (int i = 0; i != 1000; ++i)
        {
            uniform_real_distribution<float> dist(-10000, 10000);
            vector<float> source;
            source.resize(n);
            for (auto& e : source)
                e = dist(engine);
            vector<uint8_t> destination;
            destination.resize(VectorCodec::UpperBound(n));
            auto k = VectorCodec::Encode(source.data(), source.size(), destination.data());
            if (k > destination.size())
                return -1;
            destination.resize(k);
            vector<float> check;
            check.resize(source.size());
            VectorCodec::Decode(destination.data(), check.size(), check.data());
            for (size_t j = 0; j != check.size(); ++j)
                if (check[j] != source[j])
                    return -2;
        }   
    }
    for (int n = 16; n != 65536; n *= 2)
    {
        for (int i = 0; i != 1000; ++i)
        {
            uniform_real_distribution<float> dist(-10000, 10000);
            vector<float> source;
            source.resize(n);
            for (auto& e : source)
                e = dist(engine);
            vector<uint8_t> destination;
            destination.resize(VectorCodec::UpperBound(n));
            auto k = VectorCodec::EncodeQuick(source.data(), source.size(), destination.data());
            if (k > destination.size())
                return -1;
            destination.resize(k);
            vector<float> check;
            check.resize(source.size());
            VectorCodec::DecodeQuick(destination.data(), check.size(), check.data());
            for (size_t j = 0; j != check.size(); ++j)
                if (check[j] != source[j])
                    return -2;
        }   
    }
    return 0;
}