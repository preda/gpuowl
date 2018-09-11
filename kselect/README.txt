1. build kselect with "make".
2. generate the list of pairs (prime, order) with Pari-gp and prime-order.gp, e.g:
   $ gp -q prime-order.pg
   This produces the file p-order.txt
3. run kselect like this:
   $ ./kselect < p-order.txt > set.txt

This way you obtain a list of 1.8M iteration numbers "k" that are good for P-1 trials.
See https://mersenneforum.org/showpost.php?p=495794&postcount=43 for context.
