f = fileopen("p-order.txt", "w")
forprime(p=3, 500000000, z = znorder(Mod(2,p)); if(z<90000000, filewrite(f, Strprintf("%u %u", p, z))))
fileclose(f)
