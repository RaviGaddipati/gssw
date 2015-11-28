/* The MIT License

   Copyright (c) 2012-1015 Boston College.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/* Contact: Mengyao Zhao <zhangmp@bc.edu> */
/* Contact: Erik Garrison <erik.garrison@bc.edu> */

/*
 *  ssw.c
 *
 *  Created by Mengyao Zhao on 6/22/10.
 *  Copyright 2010 Boston College. All rights reserved.
 *	Version 0.1.4
 *	Last revision by Erik Garrison 01/02/2014
 *
 *	Modified by Ravi Gaddipati to remove traceback.
 *
 */

#include <emmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "gssw.h"

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch. */
__m128i *gssw_qP_byte(const int8_t *read_num,
                      const int8_t *mat,
                      const int32_t readLen,
                      const int32_t n,    /* the edge length of the squre matrix mat */
                      uint8_t bias) {

  int32_t segLen = (readLen + 15) / 16; /* Split the 128 bit register into 16 pieces.
								     Each piece is 8 bit. Split the read into 16 segments.
								     Calculat 16 segments in parallel.
								   */
  __m128i *vProfile = (__m128i *) malloc(n * segLen * sizeof(__m128i));
  int8_t *t = (int8_t *) vProfile;
  int32_t nt, i, j, segNum;

  /* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
  for (nt = 0; LIKELY(nt < n); nt++) {
    for (i = 0; i < segLen; i++) {
      j = i;
      for (segNum = 0; LIKELY(segNum < 16); segNum++) {
        *t++ = j >= readLen ? bias : mat[nt * n + read_num[j]] + bias;
        j += segLen;
      }
    }
  }
  return vProfile;
}

/* To determine the maximum values within each vector, rather than between vectors. */

#define m128i_max16(m, vm) \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 8)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 4)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 2)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 1)); \
    (m) = _mm_extract_epi16((vm), 0)

#define m128i_max8(m, vm) \
    (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 8)); \
    (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 4)); \
    (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 2)); \
    (m) = _mm_extract_epi16((vm), 0)

/* Striped Smith-Waterman
   Record the highest score of each reference position.
   Return the alignment score and ending position of the best alignment, 2nd best alignment, etc.
   Gap begin and gap extension are different.
   wight_match > 0, all other weights < 0.
   The returned positions are 0-based.
 */
gssw_alignment_end *gssw_sw_sse2_byte(const int8_t *ref,
                                      int8_t ref_dir,    // 0: forward ref; 1: reverse ref
                                      int32_t refLen,
                                      int32_t readLen,
                                      const uint8_t weight_gapO, /* will be used as - */
                                      const uint8_t weight_gapE, /* will be used as - */
                                      __m128i *vProfile,
                                      uint8_t terminate,    /* the best alignment score: used to terminate
                                                               the matrix calculation when locating the
                                                               alignment beginning point. If this score
                                                               is set to 0, it will not be used */
                                      uint8_t bias,  /* Shift 0 point to a positive value. */
                                      int32_t maskLen,
                                      gssw_align *alignment, /* to save seed and matrix */
                                      const gssw_seed *seed,
                                      __m128i *pvHStore, __m128i *pvHLoad, __m128i *pvHmax,
                                      __m128i *pvE) {     /* to seed the alignment */

  uint8_t max = 0;                             /* the max alignment score */
  int32_t end_read = readLen - 1;
  int32_t end_ref = -1; /* 0_based best alignment ending point; Initialized as isn't aligned -1. */
  uint32_t segLen = (readLen + 15) / 16; /* number of segment */


  /* Note use of aligned memory.  Return value of 0 means success for posix_memalign. */
  // Check if memory was allocated
  if (&alignment->seed.seglen == 0) {
    if (!(!posix_memalign((void **) &alignment->seed.pvE, sizeof(__m128i), segLen * sizeof(__m128i)) &&
        !posix_memalign((void **) &alignment->seed.pvHStore, sizeof(__m128i), segLen * sizeof(__m128i))
    )) {
      fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment buffers.\n");
      exit(1);
    }
  } else {
    // If it was, reset the alignment. Also does memset on the seed
    gssw_align_reset(alignment, segLen);
  }

  /* Workaround because we don't have an aligned calloc */
  memset(pvHStore, 0, segLen * sizeof(__m128i));
  memset(pvHLoad, 0, segLen * sizeof(__m128i));
  memset(pvHmax, 0, segLen * sizeof(__m128i));
  memset(pvE, 0, segLen * sizeof(__m128i));
  // memset(alignment->seed.pvE, 0, segLen * sizeof(__m128i));
  // memset(alignment->seed.pvHStore, 0, segLen * sizeof(__m128i));

  /* if we are running a seeded alignment, copy over the seeds */
  if (seed) {
    memcpy(pvE, seed->pvE, segLen * sizeof(__m128i));
    memcpy(pvHStore, seed->pvHStore, segLen * sizeof(__m128i));
  }

  /* Record that we have done a byte-order alignment */
  alignment->is_byte = 1;

  /* Define 16 byte 0 vector. */
  __m128i vZero = _mm_set1_epi32(0);

  /* Used for iteration */
  int32_t i, j;

  /* 16 byte insertion begin vector */
  __m128i vGapO = _mm_set1_epi8(weight_gapO);

  /* 16 byte insertion extension vector */
  __m128i vGapE = _mm_set1_epi8(weight_gapE);

  /* 16 byte bias vector */
  __m128i vBias = _mm_set1_epi8(bias);

  __m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
  __m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
  __m128i vTemp;
  int32_t begin = 0, end = refLen, step = 1;

  /* outer loop to process the reference sequence */
  if (ref_dir == 1) {
    begin = refLen - 1;
    end = -1;
    step = -1;
  }
#if DEBUG > 5
  uint64_t alignStart = get_timestamp();
  uint64_t lazyFt = 0;
  uint64_t temp;
#endif
  for (i = begin; LIKELY(i != end); i += step) {
    int32_t cmp;
    __m128i e = vZero, vF = vZero, vMaxColumn = vZero; /* Initialize F value to 0.
							   Any errors to vH values will be corrected in the Lazy_F loop.
							 */

    //__m128i vH = pvHStore[segLen - 1];
    __m128i vH = _mm_load_si128(pvHStore + (segLen - 1));
    vH = _mm_slli_si128(vH, 1); /* Shift the 128-bit value in vH left by 1 byte. */
    __m128i *vP = vProfile + ref[i] * segLen; /* Right part of the vProfile */

    /* Swap the 2 H buffers. */
    __m128i *pv = pvHLoad;
    pvHLoad = pvHStore;
    pvHStore = pv;

    /* inner loop to process the query sequence */
    for (j = 0; LIKELY(j < segLen); ++j) {

      vH = _mm_adds_epu8(vH, _mm_load_si128(vP + j));
      vH = _mm_subs_epu8(vH, vBias); /* vH will be always > 0 */

      /* Get max from vH, vE and vF. */
      e = _mm_load_si128(pvE + j);
      //_mm_store_si128(vE + j, e);

      vH = _mm_max_epu8(vH, e);
      vH = _mm_max_epu8(vH, vF);
      vMaxColumn = _mm_max_epu8(vMaxColumn, vH);


      /* Save vH values. */
      _mm_store_si128(pvHStore + j, vH);

      /* Update vE value. */
      vH = _mm_subs_epu8(vH, vGapO); /* saturation arithmetic, result >= 0 */
      e = _mm_subs_epu8(e, vGapE);
      e = _mm_max_epu8(e, vH);

      /* Update vF value. */
      vF = _mm_subs_epu8(vF, vGapE);
      vF = _mm_max_epu8(vF, vH);

      /* Save E */
      _mm_store_si128(pvE + j, e);

      /* Load the next vH. */
      vH = _mm_load_si128(pvHLoad + j);
    }

#if DEBUG > 5
    temp = get_timestamp();
#endif
    /* Lazy_F loop: has been revised to disallow adjecent insertion and then deletion, so don't update E(i, j), learn from SWPS3 */
    /* reset pointers to the start of the saved data */
    j = 0;
    vH = _mm_load_si128(pvHStore + j);

    /*  the computed vF value is for the given column.  since */
    /*  we are at the end, we need to shift the vF value over */
    /*  to the next column. */
    vF = _mm_slli_si128(vF, 1);

    vTemp = _mm_subs_epu8(vH, vGapO);
    vTemp = _mm_subs_epu8(vF, vTemp);
    vTemp = _mm_cmpeq_epi8(vTemp, vZero);
    cmp = _mm_movemask_epi8(vTemp);

    while (cmp != 0xffff) {
      vH = _mm_max_epu8(vH, vF);
      vMaxColumn = _mm_max_epu8(vMaxColumn, vH);
      _mm_store_si128(pvHStore + j, vH);

      vF = _mm_subs_epu8(vF, vGapE);

      j++;
      if (j >= segLen) {
        j = 0;
        vF = _mm_slli_si128(vF, 1);
      }

      vH = _mm_load_si128(pvHStore + j);
      vTemp = _mm_subs_epu8(vH, vGapO);
      vTemp = _mm_subs_epu8(vF, vTemp);
      vTemp = _mm_cmpeq_epi8(vTemp, vZero);
      cmp = _mm_movemask_epi8(vTemp);
    }
#if DEBUG > 5
    lazyFt += get_timestamp() - temp;
#endif
    vMaxScore = _mm_max_epu8(vMaxScore, vMaxColumn);
    vTemp = _mm_cmpeq_epi8(vMaxMark, vMaxScore);
    cmp = _mm_movemask_epi8(vTemp);
    if (cmp != 0xffff) {
      uint8_t temp;
      vMaxMark = vMaxScore;
      m128i_max16(temp, vMaxScore);
      vMaxScore = vMaxMark;

      if (LIKELY(temp > max)) {
        max = temp;
        if (max + bias >= 255) break;    //overflow
        end_ref = i;

        /* Store the column with the highest alignment score in order to trace the alignment ending position on read. */
        for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];

      }
    }

  }
#if DEBUG > 5
  uint64_t alignEnd = get_timestamp();
  fprintf(stdout, "Total time (us): %llu, lazy F: %f%%\n", alignEnd - alignStart,
          100.0f * lazyFt / (alignEnd - alignStart));
#endif

  // save the last vH
  memcpy(alignment->seed.pvE, pvE, segLen * sizeof(__m128i));
  memcpy(alignment->seed.pvHStore, pvHStore, segLen * sizeof(__m128i));

  /* Trace the alignment ending position on read. */
  uint8_t *t = (uint8_t *) pvHmax;
  int32_t column_len = segLen * 16;
  for (i = 0; LIKELY(i < column_len); ++i, ++t) {
    int32_t temp;
    if (*t == max) {
      temp = i / 16 + i % 16 * segLen;
      if (temp < end_read) end_read = temp;
    }
  }

  /* Find the most possible 2nd best alignment. */
  gssw_alignment_end *bests = (gssw_alignment_end *) calloc(2, sizeof(gssw_alignment_end));
  bests[0].score = max + bias >= 255 ? 255 : max;
  bests[0].ref = end_ref;
  bests[0].read = end_read;


  return bests;
}

__m128i *gssw_qP_word(const int8_t *read_num,
                      const int8_t *mat,
                      const int32_t readLen,
                      const int32_t n) {

  int32_t segLen = (readLen + 7) / 8;
  __m128i *vProfile = (__m128i *) malloc(n * segLen * sizeof(__m128i));
  int16_t *t = (int16_t *) vProfile;
  int32_t nt, i, j;
  int32_t segNum;

  /* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
  for (nt = 0; LIKELY(nt < n); nt++) {
    for (i = 0; i < segLen; i++) {
      j = i;
      for (segNum = 0; LIKELY(segNum < 8); segNum++) {
        *t++ = j >= readLen ? 0 : mat[nt * n + read_num[j]];
        j += segLen;
      }
    }
  }
  return vProfile;
}

gssw_alignment_end *gssw_sw_sse2_word(const int8_t *ref,
                                      int8_t ref_dir,    // 0: forward ref; 1: reverse ref
                                      int32_t refLen,
                                      int32_t readLen,
                                      const uint8_t weight_gapO, /* will be used as - */
                                      const uint8_t weight_gapE, /* will be used as - */
                                      __m128i *vProfile,
                                      uint16_t terminate,
                                      int32_t maskLen,
                                      gssw_align *alignment, /* to save seed and matrix */
                                      const gssw_seed *seed,
                                      __m128i *pvHStore, __m128i *pvHLoad, __m128i *pvHmax,
                                      __m128i *pvE) {     /* to seed the alignment */


  uint16_t max = 0;                             /* the max alignment score */
  int32_t end_read = readLen - 1;
  int32_t end_ref = 0; /* 1_based best alignment ending point; Initialized as isn't aligned - 0. */
  uint32_t segLen = (readLen + 7) / 8; /* number of segment */

  /* Note use of aligned memory.  Return value of 0 means success for posix_memalign. */
  // Check if memory was allocated
  if (&alignment->seed.seglen == 0) {
    if (!(!posix_memalign((void **) &alignment->seed.pvE, sizeof(__m128i), segLen * sizeof(__m128i)) &&
        !posix_memalign((void **) &alignment->seed.pvHStore, sizeof(__m128i), segLen * sizeof(__m128i))
    )) {
      fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment buffers.\n");
      exit(1);
    }
  } else {
    // If it was, reset the alignment. Also does memset on the seed
    gssw_align_reset(alignment, segLen);
  }
  /* Workaround because we don't have an aligned calloc */
  memset(pvHStore, 0, segLen * sizeof(__m128i));
  memset(pvHLoad, 0, segLen * sizeof(__m128i));
  memset(pvHmax, 0, segLen * sizeof(__m128i));
  memset(pvE, 0, segLen * sizeof(__m128i));
  // memset(alignment->seed.pvE, 0, segLen * sizeof(__m128i));
  // memset(alignment->seed.pvHStore, 0, segLen * sizeof(__m128i));

  /* if we are running a seeded alignment, copy over the seeds */
  if (seed) {
    memcpy(pvE, seed->pvE, segLen * sizeof(__m128i));
    memcpy(pvHStore, seed->pvHStore, segLen * sizeof(__m128i));
  }


  /* Record that we have done a word-order alignment */
  alignment->is_byte = 0;

  /* Define 16 byte 0 vector. */
  __m128i vZero = _mm_set1_epi32(0);

  /* Used for iteration */
  int32_t i, j, k;

  /* 16 byte insertion begin vector */
  __m128i vGapO = _mm_set1_epi16(weight_gapO);

  /* 16 byte insertion extension vector */
  __m128i vGapE = _mm_set1_epi16(weight_gapE);

  /* 16 byte bias vector */
  __m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
  __m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
  __m128i vTemp;
  int32_t begin = 0, end = refLen, step = 1;

  /* outer loop to process the reference sequence */
  if (ref_dir == 1) {
    begin = refLen - 1;
    end = -1;
    step = -1;
  }
  for (i = begin; LIKELY(i != end); i += step) {
    int32_t cmp;
    __m128i e = vZero, vF = vZero; /* Initialize F value to 0.
							   Any errors to vH values will be corrected in the Lazy_F loop.
							 */
    __m128i vH = pvHStore[segLen - 1];
    vH = _mm_slli_si128(vH, 2); /* Shift the 128-bit value in vH left by 2 byte. */

    /* Swap the 2 H buffers. */
    __m128i *pv = pvHLoad;

    __m128i vMaxColumn = vZero; /* vMaxColumn is used to record the max values of column i. */

    __m128i *vP = vProfile + ref[i] * segLen; /* Right part of the vProfile */
    pvHLoad = pvHStore;
    pvHStore = pv;

    /* inner loop to process the query sequence */
    for (j = 0; LIKELY(j < segLen); j++) {
      vH = _mm_adds_epi16(vH, _mm_load_si128(vP + j));

      /* Get max from vH, vE and vF. */
      e = _mm_load_si128(pvE + j);
      vH = _mm_max_epi16(vH, e);
      vH = _mm_max_epi16(vH, vF);
      vMaxColumn = _mm_max_epi16(vMaxColumn, vH);

      /* Save vH values. */
      _mm_store_si128(pvHStore + j, vH);

      /* Update vE value. */
      vH = _mm_subs_epu16(vH, vGapO); /* saturation arithmetic, result >= 0 */
      e = _mm_subs_epu16(e, vGapE);
      e = _mm_max_epi16(e, vH);
      _mm_store_si128(pvE + j, e);

      /* Update vF value. */
      vF = _mm_subs_epu16(vF, vGapE);
      vF = _mm_max_epi16(vF, vH);

      /* Load the next vH. */
      vH = _mm_load_si128(pvHLoad + j);
    }

    /* Lazy_F loop: has been revised to disallow adjecent insertion and then deletion, so don't update E(i, j), learn from SWPS3 */
    for (k = 0; LIKELY(k < 8); ++k) {
      vF = _mm_slli_si128(vF, 2);
      for (j = 0; LIKELY(j < segLen); ++j) {
        vH = _mm_load_si128(pvHStore + j);
        vH = _mm_max_epi16(vH, vF);
        _mm_store_si128(pvHStore + j, vH);
        vH = _mm_subs_epu16(vH, vGapO);
        vF = _mm_subs_epu16(vF, vGapE);
        if (UNLIKELY(!_mm_movemask_epi8(_mm_cmpgt_epi16(vF, vH)))) goto end;
      }
    }

    end:
    vMaxScore = _mm_max_epi16(vMaxScore, vMaxColumn);
    vTemp = _mm_cmpeq_epi16(vMaxMark, vMaxScore);
    cmp = _mm_movemask_epi8(vTemp);
    if (cmp != 0xffff) {
      uint16_t temp;
      vMaxMark = vMaxScore;
      m128i_max8(temp, vMaxScore);
      vMaxScore = vMaxMark;

      if (LIKELY(temp > max)) {
        max = temp;
        end_ref = i;
        for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];
      }
    }
  }

  memcpy(alignment->seed.pvE, pvE, segLen * sizeof(__m128i));
  memcpy(alignment->seed.pvHStore, pvHStore, segLen * sizeof(__m128i));


  /* Trace the alignment ending position on read. */
  uint16_t *t = (uint16_t *) pvHmax;
  int32_t column_len = segLen * 8;
  for (i = 0; LIKELY(i < column_len); ++i, ++t) {
    int32_t temp;
    if (*t == max) {
      temp = i / 8 + i % 8 * segLen;
      if (temp < end_read) end_read = temp;
    }
  }

  /* Find the most possible 2nd best alignment. */
  gssw_alignment_end *bests = (gssw_alignment_end *) calloc(2, sizeof(gssw_alignment_end));
  bests[0].score = max;
  bests[0].ref = end_ref;
  bests[0].read = end_read;

  return bests;
}

gssw_profile *gssw_init(const int8_t *read, const int32_t readLen, const int8_t *mat, const int32_t n,
                        const int8_t score_size) {
  gssw_profile *p = (gssw_profile *) calloc(1, sizeof(struct gssw_profile));
  p->profile_byte = 0;
  p->profile_word = 0;
  p->bias = 0;

  if (score_size == 0 || score_size == 2) {
    /* Find the bias to use in the substitution matrix */
    int32_t bias = 0, i;
    for (i = 0; i < n * n; i++) if (mat[i] < bias) bias = mat[i];
    bias = abs(bias);

    p->bias = bias;
    p->profile_byte = gssw_qP_byte(read, mat, readLen, n, bias);
  }
  if (score_size == 1 || score_size == 2) p->profile_word = gssw_qP_word(read, mat, readLen, n);
  p->read = read;
  p->mat = mat;
  p->readLen = readLen;
  p->n = n;
  return p;
}

gssw_align *gssw_align_create(void) {
  gssw_align *a = (gssw_align *) calloc(1, sizeof(gssw_align));
  a->seed.pvHStore = NULL;
  a->seed.pvE = NULL;
  return a;
}

void gssw_align_destroy(gssw_align *a) {
  gssw_align_clear_matrix_and_seed(a);
  free(a);
}

void gssw_align_reset(gssw_align *a, uint32_t seglen) {
  if (a->seed.seglen == seglen) {
    memset(a->seed.pvE, 0, seglen * sizeof(__m128i));
    memset(a->seed.pvHStore, 0, seglen * sizeof(__m128i));

  } else {
    gssw_align_clear_matrix_and_seed(a);
    if (!(!posix_memalign((void **) &a->seed.pvE, sizeof(__m128i), seglen * sizeof(__m128i)) &&
        !posix_memalign((void **) &a->seed.pvHStore, sizeof(__m128i), seglen * sizeof(__m128i))
    )) {
      fprintf(stderr, "error:gssw_align Could not allocate memory required for alignment buffers.\n");
      exit(1);
    }
    memset(a->seed.pvE, 0, seglen * sizeof(__m128i));
    memset(a->seed.pvHStore, 0, seglen * sizeof(__m128i));
    a->seed.seglen = seglen;
  }
}

void gssw_align_clear_matrix_and_seed(gssw_align *a) {
  free(a->seed.pvHStore);
  a->seed.pvHStore = NULL;
  free(a->seed.pvE);
  a->seed.pvE = NULL;
}

void gssw_seed_destroy(gssw_seed *s) {
  free(s->pvE);
  s->pvE = NULL;
  free(s->pvHStore);
  s->pvE = NULL;
  free(s);
}

gssw_node *gssw_node_create(int data,
                            const uint32_t id,
                            const char *seq,
                            const int8_t *nt_table,
                            const int8_t *score_matrix) {
  gssw_node *n = calloc(1, sizeof(gssw_node));
  int32_t len = strlen(seq);
  n->id = id;
  n->len = len;
  n->seq = (char *) malloc(len + 1);
  strncpy(n->seq, seq, len);
  n->seq[len] = 0;
  n->data = data;
  n->num = gssw_create_num(seq, len, nt_table);
  n->count_prev = 0; // are these be set == 0 by calloc?
  n->count_next = 0;
  n->alignment = NULL;
  n->indivSize = 0;
  return n;
}


void gssw_profile_destroy(gssw_profile *prof) {
  free(prof->profile_byte);
  free(prof->profile_word);
  free(prof);
}

void gssw_node_destroy(gssw_node *n) {
  free(n->seq);
  free(n->num);
  free(n->prev);
  free(n->next);
  if (n->alignment) {
    gssw_align_destroy(n->alignment);
  }
  free(n);
}


void gssw_node_add_prev(gssw_node *n, gssw_node *m) {
  ++n->count_prev;
  n->prev = (gssw_node **) realloc(n->prev, n->count_prev * sizeof(gssw_node *));
  n->prev[n->count_prev - 1] = m;
}

void gssw_node_add_next(gssw_node *n, gssw_node *m) {
  ++n->count_next;
  n->next = (gssw_node **) realloc(n->next, n->count_next * sizeof(gssw_node *));
  n->next[n->count_next - 1] = m;
}

void gssw_nodes_add_edge(gssw_node *n, gssw_node *m) {
  // check that there isn't already an edge
  uint32_t k;
  // check to see if there is an edge from n -> m, and exit if so
  for (k = 0; k < n->count_next; ++k) {
    if (n->next[k] == m) {
      return;
    }
  }
  gssw_node_add_next(n, m);
  gssw_node_add_prev(m, n);
}

void gssw_node_add_indivs(gssw_node *n, int16_t indiv) {
  n->indivSize++;
  n->indiv = (int16_t *) realloc(n->indiv, n->indivSize * sizeof(int16_t));
  n->indiv[n->indivSize - 1] = indiv;
}


gssw_seed *gssw_create_seed_byte(int32_t readLen, gssw_node **prev, int32_t count) {
  int32_t j = 0, k = 0;
  for (k = 0; k < count; ++k) {
    if (!prev[k]->alignment) {
      fprintf(stderr, "cannot align because node predecessors cannot provide seed\n");
      fprintf(stderr, "failing is node %u\n", prev[k]->id);
      exit(1);
    }
  }

  __m128i vZero = _mm_set1_epi32(0);
  int32_t segLen = (readLen + 15) / 16;
  gssw_seed *seed = (gssw_seed *) calloc(1, sizeof(gssw_seed));
  if (!(!posix_memalign((void **) &seed->pvE, sizeof(__m128i), segLen * sizeof(__m128i)) &&
      !posix_memalign((void **) &seed->pvHStore, sizeof(__m128i), segLen * sizeof(__m128i)))) {
    fprintf(stderr, "error:[gssw] Could not allocate memory for alignment seed\n");
    exit(1);
  }
  memset(seed->pvE, 0, segLen * sizeof(__m128i));
  memset(seed->pvHStore, 0, segLen * sizeof(__m128i));
  // take the max of all inputs
  __m128i pvE = vZero, pvH = vZero, ovE = vZero, ovH = vZero;
  for (j = 0; j < segLen; ++j) {
    pvE = vZero;
    pvH = vZero;
    for (k = 0; k < count; ++k) {
      ovE = _mm_load_si128(prev[k]->alignment->seed.pvE + j);
      ovH = _mm_load_si128(prev[k]->alignment->seed.pvHStore + j);
      pvE = _mm_max_epu8(pvE, ovE);
      pvH = _mm_max_epu8(pvH, ovH);
    }
    _mm_store_si128(seed->pvHStore + j, pvH);
    _mm_store_si128(seed->pvE + j, pvE);
  }
  return seed;
}

gssw_seed *gssw_create_seed_word(int32_t readLen, gssw_node **prev, int32_t count) {
  int32_t j = 0, k = 0;
  __m128i vZero = _mm_set1_epi32(0);
  int32_t segLen = (readLen + 7) / 8;
  gssw_seed *seed = (gssw_seed *) calloc(1, sizeof(gssw_seed));
  if (!(!posix_memalign((void **) &seed->pvE, sizeof(__m128i), segLen * sizeof(__m128i)) &&
      !posix_memalign((void **) &seed->pvHStore, sizeof(__m128i), segLen * sizeof(__m128i)))) {
    fprintf(stderr, "error:[gssw] Could not allocate memory for alignment seed\n");
    exit(1);
  }
  memset(seed->pvE, 0, segLen * sizeof(__m128i));
  memset(seed->pvHStore, 0, segLen * sizeof(__m128i));
  // take the max of all inputs
  __m128i pvE = vZero, pvH = vZero, ovE = vZero, ovH = vZero;
  for (j = 0; j < segLen; ++j) {
    pvE = vZero;
    pvH = vZero;
    for (k = 0; k < count; ++k) {
      ovE = _mm_load_si128(prev[k]->alignment->seed.pvE + j);
      ovH = _mm_load_si128(prev[k]->alignment->seed.pvHStore + j);
      pvE = _mm_max_epu16(pvE, ovE);
      pvH = _mm_max_epu16(pvH, ovH);
    }
    _mm_store_si128(seed->pvHStore + j, pvH);
    _mm_store_si128(seed->pvE + j, pvE);
  }
  return seed;
}


gssw_graph *
gssw_graph_fill(gssw_graph *graph,
                const char *read_seq,
                const int8_t *nt_table,
                const int8_t *score_matrix,
                const uint8_t weight_gapO,
                const uint8_t weight_gapE,
                const int32_t maskLen,
                const int8_t score_size,
                const uint32_t readOriginPos) {
#if DEBUG > 3
  uint64_t ggfStart = get_timestamp();
#endif


  __m128i *pvHStore;
  __m128i *pvHLoad;
  __m128i *pvHmax;
  __m128i *pvE;

  int32_t read_length = (int32_t)strlen(read_seq);
  int8_t *read_num = gssw_create_num(read_seq, read_length, nt_table);
  gssw_profile *prof = gssw_init(read_num, read_length, score_matrix, 5, score_size);
  gssw_seed *seed = NULL;
  uint16_t max_score = 0;

  int32_t segLen = (prof->readLen + 15) / 16;

  graph->maxCount = 0;
  graph->submaxCount = 0;
  graph->max_node = NULL;
  graph->submax_node = NULL;

  if (!(!posix_memalign((void **) &pvHStore, sizeof(__m128i), segLen * sizeof(__m128i)) &&
      !posix_memalign((void **) &pvHLoad, sizeof(__m128i), segLen * sizeof(__m128i)) &&
      !posix_memalign((void **) &pvHmax, sizeof(__m128i), segLen * sizeof(__m128i)) &&
      !posix_memalign((void **) &pvE, sizeof(__m128i), segLen * sizeof(__m128i)))) {
    fprintf(stderr, "Error allocating memory in graph fill\n");
    exit(1);
  }

  // for each node, from start to finish in the partial order (which should be sorted topologically)
  // generate a seed from input nodes or use existing (e.g. for subgraph traversal here)
  uint32_t i;
  gssw_node **npp = &graph->nodes[0];
  for (i = 0; i < graph->size; ++i, ++npp) {
    gssw_node *n = *npp;
    // get seed from parents (max of multiple inputs)
    if (prof->profile_byte) {
      seed = gssw_create_seed_byte(prof->readLen, n->prev, n->count_prev);
    } else {
      seed = gssw_create_seed_word(prof->readLen, n->prev, n->count_prev);
    }
    gssw_node *filled_node = gssw_node_fill(n, prof, weight_gapO, weight_gapE, maskLen, seed, pvHStore, pvHLoad,
                                            pvHmax, pvE);
    gssw_seed_destroy(seed);
    seed = NULL; // cleanup seed
    // test if we have exceeded the score dynamic range
    if (prof->profile_byte && !filled_node) {
      free(prof->profile_byte);
      prof->profile_byte = NULL;
      free(read_num);
      gssw_profile_destroy(prof);
      return gssw_graph_fill(graph,
                             read_seq,
                             nt_table,
                             score_matrix,
                             weight_gapO,
                             weight_gapE,
                             maskLen,
                             1,
                             readOriginPos);
    } else {

      /** Absolute positions **/
      uint32_t absRefEndPos = (uint32_t) n->data + 1 - n->len + n->alignment->ref_end;
      uint32_t absOptEndPos = graph->max_node ? (uint32_t) graph->max_node->data + 1 - graph->max_node->len
          + graph->max_node->alignment->ref_end : 0;
      uint32_t absSuboptEndPos = graph->submax_node ?
                                 (uint32_t) graph->submax_node->data + 1 - graph->submax_node->len
                                     + graph->submax_node->alignment->ref_end : 0;

      /** New high score found **/
      if (!graph->max_node || n->alignment->score > max_score) {
        graph->max_node = n;
        max_score = n->alignment->score;
        graph->maxCount = 1;
      }

      /** If a repeat of the max score is found away from the current max score **/
      //TODO Maybe can remove left check if the best score is guaranteed to only get higher to the right
      if ((absRefEndPos > absOptEndPos + maskLen || absRefEndPos < absOptEndPos - maskLen - prof->readLen) &&
          n->alignment->score == max_score) {
        graph->maxCount++;
        // Keep the node that's closest to the true read origin
        //TODO move best score tracking w/ actual read origin down to sw_sse2
        if (abs(readOriginPos - absRefEndPos) < abs(readOriginPos - absOptEndPos)) {
          graph->max_node = n;
          absRefEndPos = (uint32_t) n->data + 1 - n->len + n->alignment->ref_end;
          absOptEndPos =
              (uint32_t) graph->max_node->data + 1 - graph->max_node->len + graph->max_node->alignment->ref_end;
        }
      }
      /** A better suboptimal score is found, and its away from the current optimal pos **/
      if ((!graph->submax_node || n->alignment->score > graph->submax_node->alignment->score) &&
          (absRefEndPos > absOptEndPos + maskLen || absRefEndPos < absOptEndPos - maskLen - prof->readLen)) {
        graph->submax_node = n;
        graph->submaxCount = 1;
      }
        /** If a repeat suboptimal score is found away from the current suboptimal **/
      else if (graph->submax_node && n->alignment->score == graph->submax_node->alignment->score &&
          (absRefEndPos > absSuboptEndPos + maskLen || absRefEndPos < absSuboptEndPos - maskLen - prof->readLen)) {
        graph->submaxCount++;
        /** Keep the position of the suboptimal closest to the real position **/
        if (abs(readOriginPos - absRefEndPos) < abs(readOriginPos - absSuboptEndPos)) {
          graph->submax_node = n;
        }
      }

    }
  }

  free(read_num);
  gssw_profile_destroy(prof);
#if DEBUG > 3
  uint64_t ggfEnd = get_timestamp();
  fprintf(stdout, "Graph fill time (us): %llu\n", ggfEnd - ggfStart);
#endif
  free(pvHStore);
  free(pvHLoad);
  free(pvE);
  free(pvHmax);
  return graph;

}

gssw_node *
gssw_node_fill(gssw_node *node,
               const gssw_profile *prof,
               const uint8_t weight_gapO,
               const uint8_t weight_gapE,
               const int32_t maskLen,
               const gssw_seed *seed,
               __m128i *pvHStore, __m128i *pvHLoad, __m128i *pvHmax, __m128i *pvE) {

  gssw_alignment_end *bests = NULL;
  int32_t readLen = prof->readLen;

  //alignment_end* best = (alignment_end*)calloc(1, sizeof(alignment_end));
  gssw_align *alignment = node->alignment;


  if (!alignment) {
    // Create an alignment if none exists
    node->alignment = alignment = gssw_align_create();
    node->alignment->seed.seglen = 0;
  }

  // and build up a new one
  // node->alignment = alignment = gssw_align_create();


  // if we have parents, we should generate a new seed as the max of each vector
  // if one of the parents has moved into uint16_t space, we need to account for this
  // otherwise, just use the single parent alignment result as seed
  // or, if no parents, run unseeded

  // to decrease code complexity, we assume the same stripe size for the entire graph
  // this is ensured by changing the stripe size for the entire graph in graph_fill if any node scores >= 255

  // Find the alignment scores and ending positions
  if (prof->profile_byte) {
    bests = gssw_sw_sse2_byte((const int8_t *) node->num, 0, node->len, readLen, weight_gapO, weight_gapE,
                              prof->profile_byte, -1, prof->bias, maskLen, alignment, seed,
                              pvHStore, pvHLoad, pvHmax, pvE);
    if (bests[0].score == 255) {
      free(bests);
      gssw_align_clear_matrix_and_seed(alignment);
      return 0; // re-run from external context
    }
  } else if (prof->profile_word) {
    bests = gssw_sw_sse2_word((const int8_t *) node->num, 0, node->len, readLen, weight_gapO, weight_gapE,
                              prof->profile_word, -1, maskLen, alignment, seed,
                              pvHStore, pvHLoad, pvHmax, pvE);
  } else {
    fprintf(stderr, "Please call the function ssw_init before ssw_align.\n");
    return 0;
  }

  alignment->score = bests[0].score;
  alignment->ref_end = bests[0].ref;
  alignment->read_end = bests[0].read;
  free(bests);

  return node;

}

gssw_graph *gssw_graph_create(uint32_t size) {
  gssw_graph *g = calloc(1, sizeof(gssw_graph));
  g->nodes = malloc(size * sizeof(gssw_node *));
  if (!g || !g->nodes) {
    fprintf(stderr, "error:[gssw] Could not allocate memory for graph of %u nodes.\n", size);
    exit(1);
  }
  g->maxCount = 0;
  g->submaxCount = 0;
  return g;
}

void gssw_graph_destroy(gssw_graph *g) {
  uint32_t i;
  for (i = 0; i < g->size; ++i) {
    gssw_node_destroy(g->nodes[i]);
  }
  g->max_node = NULL;
  g->submax_node = NULL;
  free(g->nodes);
  g->nodes = NULL;
  free(g);
}

int32_t gssw_graph_add_node(gssw_graph *graph, gssw_node *node) {
  if (UNLIKELY(graph->size % 1024 == 0)) {
    size_t old_size = graph->size * sizeof(void *);
    size_t increment = 1024 * sizeof(void *);
    if (UNLIKELY(!(graph->nodes = realloc((void *) graph->nodes, old_size + increment)))) {
      fprintf(stderr, "error:[gssw] could not allocate memory for graph\n");
      exit(1);
    }
  }
  ++graph->size;
  graph->nodes[graph->size - 1] = node;
  return graph->size;
}

int8_t *gssw_create_num(const char *seq,
                        const int32_t len,
                        const int8_t *nt_table) {
  int32_t m;
  int8_t *num = (int8_t *) malloc(len);
  for (m = 0; m < len; ++m) num[m] = nt_table[(int) seq[m]];
  return num;
}

int8_t *gssw_create_score_matrix(int32_t match, int32_t mismatch) {
  // initialize scoring matrix for genome sequences
  //  A  C  G  T	N (or other ambiguous code)
  //  2 -2 -2 -2 	0	A
  // -2  2 -2 -2 	0	C
  // -2 -2  2 -2 	0	G
  // -2 -2 -2  2 	0	T
  //	0  0  0  0  0	N (or other ambiguous code)
  int32_t l, k, m;
  int8_t *mat = (int8_t *) calloc(25, sizeof(int8_t));
  for (l = k = 0; l < 4; ++l) {
    for (m = 0; m < 4; ++m) mat[k++] = l == m ? match : -mismatch;    /* weight_match : -weight_mismatch */
    mat[k++] = 0; // ambiguous base: no penalty
  }
  for (m = 0; m < 5; ++m) mat[k++] = 0;
  return mat;
}

int8_t *gssw_create_nt_table(void) {
  int8_t *ret_nt_table = calloc(128, sizeof(int8_t));
  int8_t nt_table[128] = {
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
  };
  memcpy(ret_nt_table, nt_table, 128 * sizeof(int8_t));
  return ret_nt_table;
}
