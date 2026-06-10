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

test_synt_valide () {
    # $1 = premier argument.
    output=$(test_synt "$1" 2>&1)
    exit_code=$?

    # exit code != 0  ou Exception ou message d'erreur standard (filename:line)
    if [ $exit_code -ne 0 ] || echo "$output" | grep -q "Exception" || echo "$output" | grep -q -e ':[0-9][0-9]*:';
    then
        echo -e "${RED}[FAIL] Erreur rencontrée sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))
    else
        echo -e "${GREEN}[OK] ${NO_COLOR}$1"
        passed_tests=$((passed_tests + 1))
    fi
}

test_synt_invalide () {
    output=$(test_synt "$1" 2>&1)
    exit_code=$?

    #if compiler crashed with Exception
    if echo "$output" | grep -q "Exception"; 
    then
        echo -e "${RED}[FAIL] Crash du compilateur sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))

    #if it failed with proper error message 
    elif [ $exit_code -ne 0 ] && echo "$output" | grep -q -e "$1:[0-9]"; 
    then
        echo -e "${GREEN}[OK] ${NO_COLOR}$1"
        passed_tests=$((passed_tests + 1))

    #if it succeeded (exit 0)
    elif [ $exit_code -eq 0 ]; 
    then
        echo -e "${RED}[FAIL] Succès inattendu sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))

    #if it failed but no standard error message found
    else
        echo -e "${RED}[FAIL] Echec non reconnu sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))
    fi
} 


echo "------------------------------"
echo -e "${BLUE}SYNTAX TESTS${NO_COLOR}"
echo "------------------------------"



# Test valid syntax files
echo "   "
echo "Execution des tests valides..."
echo "------------------------------"


for test_file in src/test/deca/syntax/valid/provided/*.deca;
do 
    test_synt_valide "$test_file";
done

for test_file in src/test/deca/syntax/valid/sans_objet/*.deca;
do
    test_synt_valide "$test_file";
done

#avec-objet
for test_file in src/test/deca/syntax/valid/avec_objet/*.deca;
do  
    test_synt_valide "$test_file";
done

#objet
for test_file in src/test/deca/syntax/valid/objet/*.deca;
do
    test_synt_valide "$test_file";
done


# Test invalid syntax files
echo "   "
echo "Execution des tests invalides..."
echo "--------------------------------"

#provided
for test_file in src/test/deca/syntax/invalid/provided/*.deca;
do 
    if test_synt "$test_file" 2>&1 | grep -q -e "$test_file:[0-9][0-9]*:"
    then
        echo -e "${GREEN}[OK] ${NO_COLOR}$test_file"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}[FAIL] Succès inattendu sur ${NO_COLOR}$test_file"
        failed_tests=$((failed_tests + 1))
    fi
done

#sans-objet
for test_file in src/test/deca/syntax/invalid/sans_objet/*.deca;
do  
    test_synt_invalide "$test_file";
done

#objet
for test_file in src/test/deca/syntax/invalid/objet/*.deca;
do  
    test_synt_invalide "$test_file";
done


echo "   "
echo "------------------------------"
echo -e "tests OK: ${GREEN}$passed_tests${NO_COLOR}; tests échoués: ${RED}$failed_tests${NO_COLOR}"