/**
 File: "Noise.hpp"

 Purpose: To generate Perlin Simplex Noise with eight iterations
 of Fractional Brownian Motion using AVX vectorized code

 Copyright (c) 2016 Michael K. Duncan (fruitless75@gmail.com)

 Distributed under the MIT License (MIT) (See accompanying file LICENSE
 or copy at http://opensource.org/licenses/MIT)
**/

#ifndef Noise_H
#define Noise_H

class Noise
{
        Noise() {}
        ~Noise() {}
        Noise(const Noise&) {}
        Noise(Noise&&) {}
        Noise& operator=(const Noise&) { return *this; }
        Noise& operator=(Noise&&) { return *this; }

    public:
        static void Seed();

        static const float getHeight();
        static const float getFrequency();
        static const float getAmplitude();
        static const float getLacunarity();
        static const float getPersistence();

        static void setHeight(const float = 1.0f);
        static void setFrequency(const float = 1.0f);
        static void setAmplitude(const float = 1.0f);
        static void setLacunarity(const float = 2.0f);
        static void setPersistence(const float = 0.5f);

        static const float getValue(const float);
        static const float getValue(const float, const float);
        static const float getValue(const float, const float, const float);
        static const float getValue(const float, const float, const float, const float);
};

#endif
