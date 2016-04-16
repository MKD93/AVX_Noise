/**
 File: "Random.hpp"

 Purpose: To generate psuedo-random numbers within
 the range of [0, 256).

 Copyright (c) 2016 Michael K. Duncan (fruitless75@gmail.com)

 Distributed under the MIT License (MIT) (See accompanying file LICENSE
 or copy at http://opensource.org/licenses/MIT)
**/

#ifndef Random_H
#define Random_H

#include <cstdint>

class Random
{
        Random() {}
        ~Random() {}
        Random(const Random&) {}
        Random(Random&&) {}
        Random& operator=(const Random&) { return *this; }
        Random& operator=(Random&&) { return *this; }

    public:
        static const uint8_t getIndex();
};

#endif
