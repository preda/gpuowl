cat <<EOM
// Copyright (C) Mihai Preda
// Generated file, do not edit. See genbundle.sh and src/cl/*.cl

#include <vector>

static const std::vector<const char*> CL_FILES{
EOM

names=

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

echo static const std::vector\<const char*\> CL_FILE_NAMES\{${names}\}\;

cat <<EOM
const std::vector<const char*>& getClFileNames() { return CL_FILES; }
const std::vector<const char*>& getClFiles() { return CL_FILE_NAMES; }
EOM
