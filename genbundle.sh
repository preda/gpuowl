cat <<EOM
// Copyright (C) Mihai Preda
// Generated file, do not edit. See genbundle.sh and src/cl/*.cl

EOM

names=

echo const char* CL_SOURCE[] = \{

for xx in $*
do
    x=`basename $xx .cl`
    
    if [ "$x" = "genbundle.sh" ] ; then continue ; fi
    
    names=${names}\"${x}\",

    echo // $xx
    #echo const char ${x}_cl[] = R\"cltag\(
    echo R\"cltag\(
    cat $xx
    echo \)cltag\"\,
    echo
done
echo \}\;

echo const char* CL_FILES[]=\{${names}\}\;
