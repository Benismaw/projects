cd "$(dirname "$0")"/../../.. || exit 1

PATH=./src/test/script/launchers:"$PATH"

# Define your colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NO_COLOR='\033[0m' # No Color (Reset)

passed_tests=0
failed_tests=0

test_context_valide () {
    output=$(test_context "$1" 2>&1)
    exit_code=$?

    # Fail if:
    # 1. The program crashed (exit_code != 0)
    # 2. An Exception was thrown (Java crash)
    # 3. A standard Deca error was found (filename:line)
    if [ $exit_code -ne 0 ] || echo "$output" | grep -q "Exception" || echo "$output" | grep -q -e "$1:[0-9]";
    then
        echo -e "${RED}[FAIL] Erreur encontré sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))
    else
        echo -e "${GREEN}[OK] ${NO_COLOR}$1"
        passed_tests=$((passed_tests + 1))
    fi
}

test_context_invalide () {
    # $1 = Le fichier à tester
    
    output=$(test_context "$1" 2>&1)
    exit_code=$?

    #if compiler crashed with Exception
    if echo "$output" | grep -q "Exception"; then
        echo -e "${RED}[FAIL] Crash du compilateur sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))

    #if it succeeded (exit 0)
    elif [ $exit_code -eq 0 ]; then
        echo -e "${RED}[FAIL] Succès inattendu (Le compilateur a accepté un code invalide) sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))

    # if it failed with proper error message (filename:line)
    elif echo "$output" | grep -q -e "$1:[0-9]"; then
        echo -e "${GREEN}[OK] ${NO_COLOR}$1"
        passed_tests=$((passed_tests + 1))

    #if it failed but no standard error message found
    else
        echo -e "${RED}[FAIL] Echec avec message inconnu sur ${NO_COLOR}$1"
        failed_tests=$((failed_tests + 1))
    fi
}


echo "------------------------------"
echo -e "${BLUE}CONTEXTE TESTS${NO_COLOR}"
echo "------------------------------"


# execute valid context tests
echo "   " 
echo "Execution des tests valides..."
echo "------------------------------"

#provided
for test_file in src/test/deca/context/valid/provided/*.deca;
do 
    test_context_valide "$test_file";
done

#sans-objet
for test_file in $(find src/test/deca/context/valid/sans_objet -name "*.deca");
do 
    test_context_valide "$test_file";
done

#OBJET
for test_file in $(find src/test/deca/context/valid/objet -name "*.deca");
do 
    test_context_valide "$test_file";
done

#not-sorted
echo "Not sorted tests:"
for test_file in src/test/deca/context/valid/not_sorted/*.deca;
do 
    test_context_valide "$test_file";
done





#execute invalid context tests
echo "   "
echo "Execution des tests invalides..."
echo "--------------------------------"   

#provided
for test_file in src/test/deca/context/invalid/provided/*.deca;
do 
    test_context_invalide "$test_file"; 
done

#sans-objet
for test_file in $(find src/test/deca/context/invalid/sans_objet -name "*.deca");
do 
    test_context_invalide "$test_file";
done

#objet
for test_file in $(find src/test/deca/context/invalid/objet -name "*.deca");
do 
    test_context_invalide "$test_file";
done    

#not-sorted
echo "Not sorted tests:"
for test_file in src/test/deca/context/invalid/not_sorted/*.deca;
do 
    test_context_invalide "$test_file";
done

echo "   "
echo "------------------------------"
echo -e "tests OK: ${GREEN}$passed_tests${NO_COLOR}; tests échoués: ${RED}$failed_tests${NO_COLOR}"

