/*  banded_sw.c
 *
 *  Created by Mengyao Zhao on 01/10/12.
 *	Version 0.1.4
 *	Last revision by Mengyao Zhao on 02/02/18.
 *
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "banded_sw.h"

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

/* Convert the coordinate in the scoring matrix into the coordinate in one line of the band. */
#define set_u(u, w, i, j) { int x=(i)-(w); x=x>0?x:0; (u)=(j)-x+1; }

/* Convert the coordinate in the direction matrix into the coordinate in one line of the band. */
#define set_d(u, w, i, j, p) { int x=(i)-(w); x=x>0?x:0; x=(j)-x; (u)=x*3+p; }

/*! @function
  @abstract  Round an integer to the next closest power-2 integer.
  @param  x  integer to be rounded (in place)
  @discussion x will be modified.
 */
#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

// Convert a positive integer to a string.
char* itoa(int32_t i) {
	char* p0 = calloc(8, sizeof(char));
	char c[8], *p1 = c, *p = p0;
	do {
    	*p1 = '0' + (i % 10);
    	i /= 10;
		++p1;
    } while (i != 0);
	do {
		--p1;
		*p = *p1;
		++p;
	}while (p1 != c);
	*p = '\0';
    return p0;
}

char* banded_sw (const char* ref, 
				 	const char* read, 
				 	int32_t refLen, 
				 	int32_t readLen,
					int32_t score,
				 	uint32_t weight_match,    /* will be used as + */
				 	uint32_t weight_mismatch, /* will be used as - */
				 	uint32_t weight_insertB,  /* will be used as - */
				 	uint32_t weight_insertE,  /* will be used as - */
				 	uint32_t weight_deletB,   /* will be used as - */
				 	uint32_t weight_deletE,   /* will be used as - */
				 	int32_t band_width,
					int8_t* nt_table,		   
				 	int8_t* mat,	/* pointer to the weight matrix */
				 	int32_t n) {	

	char* cigar = (char*)calloc(16, sizeof(char)), *p = cigar, ci = 'M';
	char* cigar1, *p1;	// reverse cigar
	int32_t i, j, e, f, temp1, temp2, s = 16, c = 0, l, max = 0;
	int32_t width, width_d, *h_b, *e_b, *h_c;
	int8_t *direction, *direction_line;

	do {
		width = band_width * 2 + 3, width_d = band_width * 2 + 1;
		h_b = (int32_t*)calloc(width, sizeof(int32_t)); 
		e_b = (int32_t*)calloc(width, sizeof(int32_t)); 
		h_c = (int32_t*)calloc(width, sizeof(int32_t)); 

		direction = (int8_t*)calloc(width_d * readLen * 3, sizeof(int8_t));
		direction_line = direction;
		for (j = 1; LIKELY(j < width - 1); j ++) h_b[j] = 0;
		for (i = 0; LIKELY(i < readLen); i ++) {
			int32_t beg = 0, end = refLen - 1, u = 0, edge;
			j = i - band_width;	beg = beg > j ? beg : j; // band start
			j = i + band_width; end = end < j ? end : j; // band end
			edge = end + 1 < width - 1 ? end + 1 : width - 1;
			f = h_b[0] = e_b[0] = h_b[edge] = e_b[edge] = h_c[0] = 0;
			direction_line = direction + width_d * i * 3;

			for (j = beg; LIKELY(j <= end); j ++) {
				int32_t b, e1, f1, d, de, df, dh;
				set_u(u, band_width, i, j);	set_u(e, band_width, i - 1, j); 
				set_u(b, band_width, i, j - 1); set_u(d, band_width, i - 1, j - 1);
				set_d(de, band_width, i, j, 0);
				set_d(df, band_width, i, j, 1);
				set_d(dh, band_width, i, j, 2);

				temp1 = i == 0 ? -weight_insertB : h_b[e] - weight_insertB;
				temp2 = i == 0 ? -weight_insertE : e_b[e] - weight_insertE;
				e_b[u] = temp1 > temp2 ? temp1 : temp2;
				direction_line[de] = temp1 > temp2 ? 3 : 2;
		
				temp1 = h_c[b] - weight_deletB;
				temp2 = f - weight_deletE;
				f = temp1 > temp2 ? temp1 : temp2;
				direction_line[df] = temp1 > temp2 ? 5 : 4;
				
				e1 = e_b[u] > 0 ? e_b[u] : 0;
				f1 = f > 0 ? f : 0;
				temp1 = e1 > f1 ? e1 : f1;
				temp2 = h_b[d] + mat[nt_table[(int)ref[j]] * n + nt_table[(int)read[i]]];
				h_c[u] = temp1 > temp2 ? temp1 : temp2;
		
				if (h_c[u] > max) max = h_c[u];
		
				if (temp1 <= temp2) direction_line[dh] = 1;
				else direction_line[dh] = e1 > f1 ? direction_line[de] : direction_line[df];
			}
			for (j = 1; j <= u; j ++) h_b[j] = h_c[j];
		}
		band_width *= 2;
	} while (LIKELY(max < score));
	band_width /= 2;

	// trace back
	i = readLen - 1;
	j = refLen - 1;
	e = 0;	// Count the number of M, D or I.
	f = 'M';
	temp2 = 2;	// h
	while (LIKELY(i > 0)) {
		set_d(temp1, band_width, i, j, temp2);
		switch (direction_line[temp1]) {
			case 1: 
				--i;
				--j;
				temp2 = 2;
				direction_line -= width_d * 3;
				f = 'M';
				break;
			case 2:
			 	--i;
				temp2 = 0;	// e
				direction_line -= width_d * 3;
				f = 'I';
				break;		
			case 3:
				--i;
				temp2 = 2;
				direction_line -= width_d * 3;
				f = 'I';
				break;
			case 4:
				--j;
				temp2 = 1;
				f = 'D';
				break;
			case 5:
				--j;
				temp2 = 2;
				f = 'D';
				break;
			default: 
				fprintf(stderr, "Trace back error: %d.\n", direction_line[temp1 - 1]);
				return 0;
		}
		if (f == ci) ++ e;
		else {
			char* num = itoa(e);
			l = strlen(num);
			c += l + 1;
			if (c >= s) {
				++s;
				kroundup32(s);
				cigar = realloc(cigar, s * sizeof(char));
				p = cigar + c - l - 1;
			}
			strcpy(p, num);
			free(num);
			p += l;
			*p = ci;
			ci = f;
			++p;
			e = 1;
		}
	}
	if (f == 'M') {
		char* num = itoa(e + 1);
		l = strlen(num);
		c += l + 1;
		if (c >= s) {
			++s;
			kroundup32(s);
			cigar = realloc(cigar, s * sizeof(char));
			p = cigar + c - l - 1;
		}
		strcpy(p, num);
		free(num);
		p += l;
		*p = 'M';
	}else {
		char* num = itoa(e);
		l = strlen(num);
		c += l + 3;	
		if (c >= s) {
			++s;
			kroundup32(s);
			cigar = realloc(cigar, s * sizeof(char));
			p = cigar + c - l - 3;
		}
		strcpy(p, num);
		free(num);
		p += l;
		*p = f;
		++p;
		*p = '1';
		++p;
		*p = 'M';
	}
	++p; *p = '\0';

	// reverse cigar
	cigar1 = (char*)calloc(strlen(cigar) + 1, sizeof(char));
	p1 = cigar1;
	l = 0;
	ci = 'M';
	p = cigar + strlen(cigar) - 1;
	while (LIKELY(p >= cigar)) {
		if (*p == 'M' || *p == 'I' || *p == 'D') {
			if (l > 0) {
				strncpy(p1, p + 1, l);
				p1 += l;
				*p1 = ci;
				++p1;
			}
			ci = *p;
			--p;
			l = 0;
		} else {
			++l;
			--p;
		}
	}
	strncpy(p1, p + 1, l);
	p1 += l;
	*p1 = ci;
	++p1;
	*p1 = '\0';

	free(direction);
	free(h_c);
	free(e_b);
	free(h_b);
	free(cigar);
	return cigar1;
}
