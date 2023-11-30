/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_ASSERT_INCLUDED
#define FAISS_ASSERT_INCLUDED

#include <cstdio>
#include <cstdlib>
#include <string>

///
/// Assertions
///

#define FAISS_ASSERT(X)                                  \
    do {                                                 \
        if (!(X)) {                                      \
            fprintf(stderr,                              \
                    "Faiss assertion '%s' failed in %s " \
                    "at %s:%d\n",                        \
                    #X,                                  \
                    __PRETTY_FUNCTION__,                 \
                    __FILE__,                            \
                    __LINE__);                           \
            abort();                                     \
        }                                                \
    } while (false)

#define FAISS_ASSERT_MSG(X, MSG)                         \
    do {                                                 \
        if (!(X)) {                                      \
            fprintf(stderr,                              \
                    "Faiss assertion '%s' failed in %s " \
                    "at %s:%d; details: " MSG "\n",      \
                    #X,                                  \
                    __PRETTY_FUNCTION__,                 \
                    __FILE__,                            \
                    __LINE__);                           \
            abort();                                     \
        }                                                \
    } while (false)

#define FAISS_ASSERT_FMT(X, FMT, ...)                    \
    do {                                                 \
        if (!(X)) {                                      \
            fprintf(stderr,                              \
                    "Faiss assertion '%s' failed in %s " \
                    "at %s:%d; details: " FMT "\n",      \
                    #X,                                  \
                    __PRETTY_FUNCTION__,                 \
                    __FILE__,                            \
                    __LINE__,                            \
                    __VA_ARGS__);                        \
            abort();                                     \
        }                                                \
    } while (false)

#endif
