/*
 Copyright (c) 2020, FUJITSU LIMITED
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the <organization> nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
==============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#define A_CONSTANT_1  48271
#define A_CONSTANT_16 1357852417
#define M_CONSTANT    0x7FFFFFFF

/*
Used multiplicative congruential generator (MCG):
Park, Stephen K.; Miller, Keith W.; Stockmeyer, Paul K. (1988).
"Technical Correspondence: Response" (PDF). Communications of the ACM. 36 (7): 108-110.
doi:10.1145/159544.376068.
*/

struct StreamStatePtr_ampl{
    int METHOD;                /* METHOD 0:MCG31 */
    uint32_t *s;               /* seed */
};

/**
* Step forward one sequence for random number generation with MCG.
*/

uint32_t next_MCG(uint32_t s[1], uint32_t A_CONSTANT) {
    s[0] = (A_CONSTANT*s[0])&M_CONSTANT;
    return s[0];
}

/**
* sve version function of next with MCG.
*/
svuint32_t  nextsve_MCG(uint32_t s[16], uint32_t A_CONSTANT) {
    int64_t j = 0;
    svbool_t pg = svwhilelt_b32_s64(j, 16);
    svuint32_t s0_sve = svld1(pg, &s[j]);
    svuint32_t u0_sve = svmul_n_u32_z(pg, s0_sve, A_CONSTANT);
    svuint32_t result0_sve = svand_z(pg, u0_sve, (uint32_t)M_CONSTANT);
    svst1(pg, &s[j], result0_sve);
    return result0_sve;
}

/**
* Preparation for sve calculation.
*
* @param[out] stream
* @param[in] METHOD 0:MCG31
* @param[in] seed The initial state for random number generation.
*/

void NewStream_ampl(
    struct StreamStatePtr_ampl *stream,
    int METHOD,
    int seed
){
    stream->METHOD = METHOD;

    // Initialization
    if(METHOD == 0/* MCG*/){
        // It is recommended to have more elements than L1 cache.
        // Cache coherency degrades performance when less than L1 cache
        stream->s=(uint32_t*)malloc(sizeof(uint32_t)*4096); // The actual number of required elements is 16.

        // Set seed
        stream->s[0] = (seed==0)?1:seed;
        for(int i=0; i < 15; i++){
            stream->s[i+1] = next_MCG(stream->s, (uint32_t)A_CONSTANT_1);
        }
        stream->s[0] = (seed==0)?1:seed;
    }else{
    }
}

/**
* To support multi-threaded computation.
*
* @param[in, out] StreamStatePtr_ampl stream
* @param[in] begin Number of jumping ahead.
*/

void SkipAheadStream_ampl(
    struct StreamStatePtr_ampl stream,
    int begin
){

    // SkipAheadStream for MCG
    if(stream.METHOD == 0/* MCG*/){

        // Set jump variables
        uint32_t total_element = begin;
        uint32_t jump_num = 21;
        uint32_t jump_constant[21] ={A_CONSTANT_1, 182605793, 1533981633, 773027713, A_CONSTANT_16, 1820286465,
                                     1065532417, 2031450113, 1516957697, 1440079873, 799784961, 1868005377,
                                     514785281, 1029570561, 2059141121, 1970798593, 1794113537, 1440743425,
                                     734003201, 1468006401, 788529153/*=A_CONSTANT_1048576*/};
        // Compute
        for(int i_jump=jump_num-1; i_jump>=0; i_jump--){
            int nloop = total_element>>i_jump;
            if(nloop==0)continue;
            for(int i = 0; i < nloop; i++){
                svuint32_t tmp = nextsve_MCG(stream.s, jump_constant[i_jump]);
            }
            total_element -= (nloop<<i_jump);
            if(total_element==0)break;
        }
    }else{
    }
}

/**
* Generation of random numbers by Bernoulli distribution.
*
* @param[out] r Value (0 or 1) genereted by the Bernoulli function.
* @param[in] METHOD 0:MCG31
* @param[in] StreamStatePtr_ampl stream
* @param[in] N Number of random numbers to generate.
* @param[in] r Array of random numbers.
* @param[in] p Parameters of the Bernoulli distribution.
* @return 0
*/
int RngBernoulli_ampl(
    int METHOD,
    struct StreamStatePtr_ampl stream,
    int N,
    int* r,
    double p
){
    int64_t i = 0;
    svuint32_t tmp;

    // MCG random generator
    if(stream.METHOD == 0/* MCG*/){
        i = 0;
        float32_t INT32_MAX_float_rec = (float32_t)(1.0/M_CONSTANT);
        svbool_t pg = svwhilelt_b32_s64(i, N);
        do{
    	    tmp = nextsve_MCG(stream.s, (uint32_t)A_CONSTANT_16);
            svfloat32_t tmp_f = svcvt_f32_z(pg, tmp);
            svfloat32_t u_sve = svmulx_z(pg, tmp_f, INT32_MAX_float_rec);
            svbool_t mask = svcmple(pg, u_sve, p);
            svint32_t r_sve = svdup_s32_z(mask, 1);
            svst1(pg, &r[i], r_sve);
            i += svcntw();
            pg = svwhilelt_b32_s64(i, N);
        }while(svptest_any(svptrue_b32(), pg));
    }else{
    }
    return 0;
}

/**
* Delete stream.
*
* @param[in] StreamStatePtr_ampl stream
*/
void DeleteStream_ampl(
    struct StreamStatePtr_ampl stream
){
    free(stream.s);
}