/**
 File: "Random.cpp"

 Purpose: To generate psuedo-random numbers within
 the range of [0, 256).

 Copyright (c) 2016 Michael K. Duncan (fruitless75@gmail.com)

 Distributed under the MIT License (MIT) (See accompanying file LICENSE
 or copy at http://opensource.org/licenses/MIT)
**/

#include "Random.hpp"

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
static HCRYPTPROV Handle = { 0 };
#else
#include <sys/stat.h>
#include <fcntl.h>
static int32_t Handle = 0;
#endif
static uint64_t Seed[2] = { 0, 0 };
static bool Start = true;

static void getSeeds()
{
    #ifdef _WIN32
    if(CryptAcquireContext(&Handle, nullptr, nullptr, PROV_RSA_FULL, 0) == static_cast<BOOL>(true))
    {
        CryptGenRandom(Handle, 16, reinterpret_cast<PBYTE>(&(Seed[0])));
        CryptReleaseContext(Handle, 0);
    }
    #else
    if((Handle = static_cast<int32_t>(open("/dev/urandom", O_RDONLY))) != -1)
    {
        read(Handle, reinterpret_cast<void*>(&(Seed[0])), 16);
        close(Handle);
    }
    #endif
}

const uint8_t Random::getIndex()
{
    if(Start == true)
    {
        getSeeds();
        Start = false;
    }

    const uint64_t value_0 = Seed[1];
    uint64_t value_1 = Seed[0];

    Seed[0] = value_0;
    value_1 ^= (value_1 << 23);
    Seed[1] = (((value_0 ^ value_1) ^ (value_0 >> 26)) ^ (value_1 >> 17));

    return static_cast<uint8_t>((Seed[1] + value_0) & 255);
}
