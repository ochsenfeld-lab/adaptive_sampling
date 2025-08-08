#include<stdlib.h>
#include<stdio.h>

#define INT int
int FILE_err=0;
INT *item_order=NULL, *item_perm=NULL, *org_frq=NULL;

typedef struct {
  INT **trsact, *buf;   // transactions, its buffer, and their sizes
  INT trsact_num, item_max, eles;  // #transactions, #items, total #elements
  INT **occ, *occ_buf, *occt;  // occurrences, its buffer, and their sizes
  INT *frq;    // array for frequency of each item, and those in the original database
  INT *itemset, itemset_siz;  // itemset array and its size
  INT *jump, jump_siz; // candidate queue and its size
  INT sigma; // minimum support
} TRSACT;

/* allocate memory with error routine */
char *alloc_memory (size_t siz){
  char *p = (char *)calloc (siz, 1);
  if ( p == NULL ){
    printf ("out of memory\n");
    exit (1);
  }
  return (p);
}

/* re-allocate memory and fill the newly allocated part by 0, with error routine */
char *realloc_memory (char *p, int sizof, size_t siz, size_t *org_siz){
  size_t i;
  p = (char *)realloc(p, siz*sizof);
  if ( p == NULL ){
    printf ("out of memory\n");
    exit (1);
  }
  for ( i=(*org_siz)*sizof ; i<siz*sizof ; i++ ) p[i] = 0;
  *org_siz = siz;
  return (p);
}

/* comparison subroutines for quick-sort */
int qsort_cmp_frq (const void *x, const void *y){
  if ( org_frq[*((INT *)x)] > org_frq[*((INT *)y)] ) return (-1);
  else return ( org_frq[*((INT *)x)] < org_frq[*((INT *)y)] );
}
int qsort_cmp_idx (const void *x, const void *y){
  if ( item_order[*((INT *)x)] < item_order[*((INT *)y)] ) return (-1);
  else return ( item_order[*((INT *)x)] > item_order[*((INT *)y)] );
}
int qsort_cmp__idx (const void *x, const void *y){
  if ( item_order[*((INT *)x)] > item_order[*((INT *)y)] ) return (-1);
  else return ( item_order[*((INT *)x)] < item_order[*((INT *)y)] );
}

/* read an integer from the file */
INT FILE_read_int (FILE *fp){
  INT item;
  int flag =1;
  int ch;
  FILE_err = 0;
  do {
    ch = fgetc (fp);
    if ( ch == '\n' ){ FILE_err = 5; return (0); }
    if ( ch < 0 ){ FILE_err = 6; return (0); }
    if ( ch=='-' ) flag = -1;
  } while ( ch<'0' || ch>'9' );
  for ( item=(int)(ch-'0') ; 1 ; item=item*10 +(int)(ch-'0') ){
    ch = fgetc (fp);
    if ( ch == '\n' ){ FILE_err = 1; return (flag*item); }
    if ( ch < 0 ){ FILE_err = 2; return (flag*item); }
    if ( (ch < '0') || (ch > '9')) return (flag*item);
  }
}

void TRSACT_print (TRSACT *T, INT *occ, INT frq, INT end){
  INT i, j, *tr, item, tID;
  printf ("------------\n");
  for ( i=0 ; i<frq ; i++ ){
    tID = occ[i];
    tr = T->trsact[tID];
    for ( j=0 ; (item=tr[j]) != end ; j++ ) printf ("%d ", item);
    printf ("\n");
  }
  printf ("============\n");
}

/* load a transaction from the input file to memory */
void TRSACT_file_load (TRSACT *T, char *fname){
  INT item, t;
  size_t cnt=0, new_cnt;
  FILE *fp = fopen (fname,"r");

  if ( !fp ){ printf ("file open error\n"); exit (1); }
  T->trsact_num = 0; // set #transactions to 0
  T->item_max = 0; // set max. index of item
  T->eles = 0;
  T->frq = NULL;

   // scan file to count the 
  do {
    do {
      item = FILE_read_int (fp);
      if ( (FILE_err&4) == 0){  // got an item from the file before reaching to line-end
        if ( item >= (INT)cnt ){
          new_cnt = cnt*2; if ( item >= (INT)new_cnt ) new_cnt = item +1;
          org_frq = (INT*)realloc_memory((char*)org_frq, sizeof(INT), new_cnt, &cnt);
        }
        if ( item >= T->item_max ) T->item_max = item+1;  // uppdate the maximum item
        org_frq[item]++;
        T->eles ++;
      }
    } while ((FILE_err&3)==0);
    T->trsact_num++;  // increase #transaction
  } while ( (FILE_err&2)==0);
  
    // compute the item ordering
  item_order = (INT *)alloc_memory ( sizeof(INT) * T->item_max );
  item_perm = (INT *)alloc_memory ( sizeof(INT) * T->item_max );
  for ( t=0 ; t<T->item_max ; t++ ) item_perm[t] = t;
  qsort (item_perm, T->item_max, sizeof(INT), qsort_cmp_frq);
  for ( t=0 ; t<T->item_max ; t++ ) item_order[item_perm[t]] = t;

    // allocate memory for transactions
  T->buf = (INT *)alloc_memory ( sizeof(INT) * (T->eles+T->trsact_num) );
  T->trsact = (INT **)alloc_memory ( sizeof(INT*) * T->trsact_num );

    // load the file
  cnt = 0;
  fseek (fp, 0, SEEK_SET);
  for ( t=0 ; t<T->trsact_num ; t++){
    T->trsact[t] = &T->buf[cnt];
    do {
      item = FILE_read_int (fp);
      if ( (FILE_err&4) == 0){ T->buf[cnt] = item; cnt++;}
    } while ((FILE_err&3)==0);
    qsort (T->trsact[t], &T->buf[cnt]-T->trsact[t], sizeof(INT), qsort_cmp_idx); // sort each transaction 
    T->buf[cnt] = T->item_max;  // attach an end mark
    cnt++;
  }

}

/* compute the occurrences of all items by occurrence deliver */
void occ_deliv (TRSACT *T, INT item){
  INT t, i, j, *tr, tID;
  for ( t=0 ; t<T->occt[item] ; t++ ){
    tID = T->occ[item][t];
    tr = T->trsact[tID];
    for ( j=0 ; (i=tr[j]) != item ; j++ ){
      if ( T->occt[i] == 0 ) T->jump[T->jump_siz++] = i;
      T->occ[i][T->occt[i]++] = tID;
    }
  }
}


/* output an itemset */
void output_itemset (TRSACT *T, INT frq){
  INT i;
  for ( i=0 ; i<T->itemset_siz ; i++) printf ("%d ", T->itemset[i]);
  printf ("(%d)\n", frq);
}

/* main recursive call */
void LCM (TRSACT *T, INT item){
  INT i, jt = T->jump_siz, flag =0;

  output_itemset (T, T->occt[item]);  // output an itemset
  occ_deliv (T, item);  // compute each item frequency
  qsort (&T->jump[jt], T->jump_siz-jt, sizeof(INT), qsort_cmp__idx);  // sort the items 

  while ( T->jump_siz > jt ){
    i = T->jump[--T->jump_siz];
    if ( T->occt[i] >= T->sigma ){
      T->itemset[T->itemset_siz++] = i;
//      if ( flag == 0 ){ flag = 1; output_itemset (T, T->occt[i]); }
//      else {
        LCM (T, i);
//      }
      T->itemset_siz--;
    }
    T->occt[i] = 0;
  }
}



/* main function, for initialization */
main (int argc, char *argv[]){
  TRSACT T;
  INT i;
  size_t cnt;
  
  if ( argc < 3 ){
    printf ("LCM: enumerate all frequent itemsets\n lcm filename support outputfile\n");
    exit(1);
  }
  T.sigma = atoi(argv[2]);
  TRSACT_file_load (&T, argv[1]);

    // initialize itemset
  T.itemset = (INT *)alloc_memory (sizeof(INT)*T.item_max);
  T.itemset_siz = 0;

    // initialize occ's
  T.occ_buf = (INT *)alloc_memory ( sizeof(INT) * (T.eles+T.trsact_num) );
  T.occ = (INT **)alloc_memory ( sizeof(INT*) * (T.item_max+1) );
  T.occt = (INT *)alloc_memory ( sizeof(INT) * (T.item_max+1) );
  for ( i=0,cnt=0 ; i<T.item_max ; i++ ){
    T.occ[i] = &T.occ_buf[cnt];
    cnt += org_frq[i];
  }
  free (org_frq);
  T.occ[T.item_max] = &T.occ_buf[cnt];
  for ( i=0 ; i<T.trsact_num ; i++ ) T.occ[T.item_max][i] = i;
  T.occt[T.item_max] = T.trsact_num;

    // initialize candidate list
  T.jump = (INT *)alloc_memory ( sizeof(INT) * T.item_max );
  T.jump_siz = 0;
  
    // main recursive call
  LCM (&T, T.item_max);
  
    // free the memory
  free (T.trsact);
  free (T.buf);
  free (T.occ);
  free (T.occt);
  free (T.occ_buf);
  free (T.jump);
  free (T.itemset);
  free (item_order);
  free (item_perm);
}



