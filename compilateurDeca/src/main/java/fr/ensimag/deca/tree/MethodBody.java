package fr.ensimag.deca.tree;


import java.io.PrintStream;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;

public class MethodBody extends AbstractMethodBody {
    private ListDeclVar variables;
    private ListInst insts;

    public MethodBody(ListDeclVar variables, ListInst insts) {
        this.variables = variables;
        this.insts = insts;
    }

    public int getNumLocals() {
        return variables.size();
    }

    protected void verifyMethodBody(DecacCompiler compiler, 
            EnvironmentExp localEnv, ClassDefinition currentClass, Type returnType) 
            throws ContextualError {
        variables.verifyListDeclVariable(compiler, localEnv, currentClass);

        insts.verifyListInst(compiler, localEnv, currentClass, returnType);
    }

    public void decompile(IndentPrintStream s) {
        s.println("{");
        variables.decompile(s);
        insts.decompile(s);
        s.print("}");
    }

    @Override
    public void codeGenMethodBody(DecacCompiler compiler) {
        // Initialisation des variables locales
        variables.codeGenListDeclVar(compiler);

        // Exécution des instructions
        insts.codeGenListInst(compiler);
    }

    protected void iterChildren(TreeFunction f) {
        variables.iter(f);
        insts.iter(f);
    }

    protected void prettyPrintChildren(PrintStream s, String prefix) {
        variables.prettyPrint(s, prefix, false);
        insts.prettyPrint(s, prefix, true);
    }
}