cd "$(dirname "$0")"/../../.. || exit 1

PATH=./src/test/script/launchers:"$PATH"

# Define your colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NO_COLOR='\033[0m'

passed_tests=0
failed_tests=0

test_lex_valide () {
    # $1 = premier argument.
    if test_lex "$1" 2>&1 | grep -q -e ':[0-9][0-9]*:'; 
    then
        echo -e "${RED}[FAIL] Erreur encontré sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))
    else
        echo -e "${GREEN}[OK] ${NO_COLOR}$1"
        passed_tests=$((passed_tests + 1))
    fi
}

test_lex_invalide () {
    # $1 = premier argument.
    if test_lex "$1" 2>&1 | grep -q -e "$1:[0-9][0-9]*:";
    then
        echo -e "${GREEN}[OK] ${NO_COLOR}$1"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}[FAIL] Succès inattendu de test_lex sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))
    fi
} 


echo "------------------------------"
echo -e "${BLUE}LEXICOGRAPHIC TESTS${NO_COLOR}"
echo "------------------------------"


# Test valid lex files
echo "   "
echo "Execution des tests valides..."
echo "------------------------------"

for test_file in src/test/deca/syntax/lex/valid/test/*.deca;
do 
    test_lex_valide "$test_file";
done


# Test invalid lex files
echo "   "
echo "Execution des tests invalides..."
echo "--------------------------------"

for test_file in src/test/deca/syntax/lex/invalid/test/*.deca;
do 
    test_lex_invalide "$test_file";
done


echo "   "
echo "------------------------------"
echo -e "tests OK: ${GREEN}$passed_tests${NO_COLOR}; tests échoués: ${RED}$failed_tests${NO_COLOR}"