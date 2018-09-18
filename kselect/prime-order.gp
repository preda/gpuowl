f = fileopen("p-order.txt", "w")
forprime(p=3000000, 2000000000, z = znorder(Mod(2,p)); if(z<90000000, filewrite(f, Strprintf("%u %u", p, z))))
fileclose(f)
